"""
Common utilities for biopb gRPC services.

Provides:
- Image encoding/decoding
- Token authentication interceptor
- Structured logging setup
- Base servicer class with error handling
"""

import json
import logging
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from typing import Optional

import biopb.image as proto
import grpc
import numpy as np
from biopb.image.utils import deserialize_to_numpy, serialize_from_numpy
from google.protobuf.json_format import MessageToDict

_AUTH_HEADER_KEY = "authorization"
_MAX_MSG_SIZE = 1024 * 1024 * 128  # 128MB

logger = logging.getLogger(__name__)


# =============================================================================
# Logging Configuration
# =============================================================================


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "latency_ms"):
            log_entry["latency_ms"] = record.latency_ms
        return json.dumps(log_entry)


def setup_logging(
    debug: bool = False,
    json_format: bool = False,
    log_level: Optional[int] = None,
) -> None:
    """
    Configure logging for biopb services.

    Args:
        debug: If True, use DEBUG level and text format (human-readable)
        json_format: If True and not debug, use JSON format for production
        log_level: Override log level (defaults to DEBUG if debug else INFO)
    """
    if log_level is None:
        log_level = logging.DEBUG if debug else logging.INFO

    # Determine format
    if debug:
        # Human-readable format for debugging
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    elif json_format:
        formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
    else:
        # Simple text format for production
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add configured handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


class RequestLogger:
    """
    Context manager for request logging with timing.

    Usage:
        with RequestLogger("RunDetection", request.ByteSize()) as log:
            # ... process request ...
            log.response_size = response.ByteSize()
    """

    def __init__(self, method: str, request_size: int = 0):
        self.method = method
        self.request_size = request_size
        self.response_size = 0
        self.start_time = time.perf_counter()
        self._logger = logging.getLogger(f"request.{method}")

    def __enter__(self):
        self._logger.debug(
            f"Request started: {self.method}, size={self.request_size} bytes"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000

        if exc_type is not None:
            self._logger.error(
                f"Request failed: {self.method}, "
                f"latency={elapsed_ms:.2f}ms, error={exc_val}"
            )
        else:
            self._logger.info(
                f"Request completed: {self.method}, "
                f"latency={elapsed_ms:.2f}ms, response_size={self.response_size} bytes"
            )

        return False  # Don't suppress exceptions


# =============================================================================
# Image Utilities
# =============================================================================


def decode_image(pixels: proto.Pixels) -> np.ndarray:
    """
    Decode protobuf Pixels to numpy array.

    Args:
        pixels: Protobuf Pixels message

    Returns:
        Numpy array with shape (T, Z, Y, X, C) or subset

    Raises:
        ValueError: If image has unsupported dimensions
    """
    if pixels.size_t > 1:
        raise ValueError("Image data has a non-singleton T dimension.")

    if pixels.size_c > 3:
        raise ValueError("Image data has more than 3 channels.")

    np_img = deserialize_to_numpy(pixels)
    return np_img


def encode_image(image: np.ndarray, **kwargs) -> proto.Pixels:
    """
    Encode numpy array to protobuf Pixels.

    Args:
        image: Numpy array
        **kwargs: Additional arguments for serialize_from_numpy

    Returns:
        Protobuf Pixels message
    """
    return serialize_from_numpy(image, **kwargs)


def parse_kwargs(request, defaults: dict) -> dict:
    """
    Merge request kwargs with defaults.

    Args:
        request: DetectionRequest or ProcessRequest with optional kwargs field
        defaults: Dictionary of default parameter values

    Returns:
        Dictionary with defaults overridden by any kwargs from the request
    """
    kwargs = defaults.copy()
    if request.HasField("kwargs"):
        request_kwargs = MessageToDict(request.kwargs)
        kwargs.update(request_kwargs)
    return kwargs


def _coerce_value(value, expected_type: str):
    """
    Attempt to coerce a value to the expected type.

    Returns (coerced_value, warning_message) where warning_message is None on success.
    Coercion rules:
    - int: Accept int, or convert float to int (truncates toward zero)
    - number: Accept int or float (Python handles this naturally)
    - bool: Only accept bool (int/float are not valid bools in protobuf struct)
    - string: Only accept string
    - array: Accept list or tuple
    """
    warning = None

    if expected_type == "int":
        if isinstance(value, bool):
            # bool is a subclass of int, but we don't want to accept it as int
            warning = f"Parameter is a boolean, expected integer"
        elif isinstance(value, int):
            pass  # Already correct type
        elif isinstance(value, float):
            # Convert float to int (struct_pb2 doesn't distinguish int/float)
            coerced = int(value)
            if coerced != value:
                warning = f"Parameter value {value} truncated to {coerced} for integer type"
            return coerced, warning
        else:
            warning = f"Parameter type {type(value).__name__} cannot be converted to integer"

    elif expected_type == "number":
        if isinstance(value, bool):
            warning = f"Parameter is a boolean, expected number"
        elif isinstance(value, (int, float)):
            pass  # Valid number type
        else:
            warning = f"Parameter type {type(value).__name__} cannot be converted to number"

    elif expected_type == "bool":
        if not isinstance(value, bool):
            warning = f"Parameter type {type(value).__name__}, expected boolean"

    elif expected_type == "string":
        if not isinstance(value, str):
            warning = f"Parameter type {type(value).__name__}, expected string"

    elif expected_type == "array":
        if isinstance(value, tuple):
            # Convert tuple to list
            return list(value), warning
        elif not isinstance(value, list):
            warning = f"Parameter type {type(value).__name__}, expected array"

    return value, warning


def _validate_type(key: str, value, expected_type: str, spec: dict) -> tuple[any, str | None]:
    """
    Validate and optionally coerce a value to the expected type.

    Args:
        key: Parameter name (for warnings)
        value: The value to validate/coerce
        expected_type: Expected type string ("int", "number", "bool", "string", "array")
        spec: Full spec dict (used for array item_type validation)

    Returns:
        Tuple of (possibly_coerced_value, error_message)
        - On success: (value or coerced_value, None)
        - On type mismatch that can be coerced: (coerced_value, None) with warning logged
        - On uncoercible type: (original_value, error_message)

    Note:
        struct_pb2 (Google protobuf Struct) does not distinguish between int and float.
        All numeric values are stored as float. This function handles that by accepting
        floats for int parameters and converting them.
    """
    # Coerce the value if needed
    coerced_value, warning = _coerce_value(value, expected_type)

    if warning:
        logger.warning(f"Parameter '{key}': {warning}")

    # For arrays, validate and coerce items
    if expected_type == "array" and isinstance(coerced_value, list):
        item_type = spec.get("item_type")
        if item_type:
            coerced_items = []
            for i, item in enumerate(coerced_value):
                item_value, item_warning = _coerce_value(item, item_type)
                if item_warning:
                    # For items, we still log warning but don't error
                    logger.warning(f"Parameter '{key}[{i}]': {item_warning}")
                coerced_items.append(item_value)
            coerced_value = coerced_items

    return coerced_value, None


def validate_kwargs(kwargs: dict, schema: dict) -> list[str]:
    """
    Validate and coerce kwargs against a schema.

    Args:
        kwargs: Dictionary of parameter values to validate (modified in-place with coerced values)
        schema: Schema dict defining valid parameters and their constraints

    Returns:
        List of error messages (empty if valid)

    Schema format:
        {
            "param_name": {
                "type": "int" | "number" | "bool" | "array" | "string",
                "item_type": "int" | "number" | ... (for arrays),
                "minimum": number (optional),
                "maximum": number (optional),
                "min_length": int (optional, for arrays),
                "max_length": int (optional, for arrays),
                "description": str (optional, for documentation),
            },
            ...
        }

    Note:
        kwargs is modified in-place with type-coerced values (e.g., float→int conversion).
        This handles struct_pb2's lack of int/float distinction.
    """
    errors = []

    # Check for unknown parameters - log warning instead of error
    # (struct_pb2 may include extra fields we don't know about)
    for key in kwargs:
        if key not in schema:
            logger.warning(f"Unknown parameter '{key}' will be ignored")

    # Validate and coerce each known parameter
    for key, spec in schema.items():
        if key not in kwargs:
            continue

        value = kwargs[key]
        expected_type = spec.get("type")

        # Type validation and coercion
        coerced_value, type_error = _validate_type(key, value, expected_type, spec)
        if type_error:
            errors.append(type_error)
            continue

        # Update kwargs with coerced value (e.g., float→int)
        kwargs[key] = coerced_value

        # Range validation for numbers (use coerced value)
        if expected_type in ("number", "int"):
            if "minimum" in spec and coerced_value < spec["minimum"]:
                errors.append(f"Parameter '{key}' value {coerced_value} is below minimum {spec['minimum']}")
            if "maximum" in spec and coerced_value > spec["maximum"]:
                errors.append(f"Parameter '{key}' value {coerced_value} exceeds maximum {spec['maximum']}")

        # Length validation for arrays
        if expected_type == "array":
            if "min_length" in spec and len(coerced_value) < spec["min_length"]:
                errors.append(f"Parameter '{key}' must have at least {spec['min_length']} items")
            if "max_length" in spec and len(coerced_value) > spec["max_length"]:
                errors.append(f"Parameter '{key}' must have at most {spec['max_length']} items")

    return errors


# =============================================================================
# Authentication
# =============================================================================


class TokenValidationInterceptor(grpc.ServerInterceptor):
    """gRPC interceptor for Bearer token authentication."""

    def __init__(self, token: Optional[str]):
        def abort(ignored_request, context):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid token signature")

        self._abort_handler = grpc.unary_unary_rpc_method_handler(abort)
        self.token = token

    def intercept_service(self, continuation, handler_call_details):
        # Allow health checks without authentication
        method = handler_call_details.method
        if method and "grpc.health.v1.Health" in method:
            return continuation(handler_call_details)

        expected_metadata = (_AUTH_HEADER_KEY, f"Bearer {self.token}")
        if (
            self.token is None
            or expected_metadata in handler_call_details.invocation_metadata
        ):
            return continuation(handler_call_details)
        else:
            return self._abort_handler


# =============================================================================
# Base Servicer
# =============================================================================


class BiopbServicerBase(proto.ObjectDetectionServicer, proto.ProcessImageServicer):
    """
    Base class for biopb servicers with error handling and logging.

    Provides:
    - Optional thread-safe request handling via lock
    - Error handling with proper gRPC status codes and correlation IDs
    - Full traceback logging for all errors

    Subclasses should implement RunDetection and Run methods.

    Args:
        use_lock: If True, serialize requests with a lock. Default True for
            backwards compatibility. Set False for thread-safe models.
    """

    def __init__(self, use_lock: bool = True):
        self._lock = threading.RLock() if use_lock else None
        self._use_lock = use_lock

    @contextmanager
    def _server_context(self, context):
        """
        Context manager for request handling with error handling.

        Usage:
            def RunDetection(self, request, context):
                with self._server_context(context):
                    # ... process request ...
                    return response
        """
        try:
            if self._use_lock:
                with self._lock:
                    yield
            else:
                yield

        # Let gRPC abort exceptions propagate (avoid double-abort)
        except grpc.RpcError:
            raise

        except ValueError as e:
            error_id = uuid.uuid4().hex[:8]
            logger.error(f"[{error_id}] Invalid argument: {e}")
            logger.error(f"[{error_id}] Traceback:\n{traceback.format_exc()}")
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"{repr(e)} (error_id: {error_id})",
            )

        except NotImplementedError as e:
            error_id = uuid.uuid4().hex[:8]
            logger.error(f"[{error_id}] Not implemented: {e}")
            logger.error(f"[{error_id}] Traceback:\n{traceback.format_exc()}")
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                f"{repr(e)} (error_id: {error_id})",
            )

        except Exception as e:
            error_id = uuid.uuid4().hex[:8]
            logger.error(f"[{error_id}] Prediction failed: {e}")
            logger.error(f"[{error_id}] Traceback:\n{traceback.format_exc()}")

            # Check for CUDA errors and log helpful message
            error_str = str(e).lower()
            if "cuda" in error_str or "gpu" in error_str:
                if "out of memory" in error_str:
                    logger.warning(
                        f"[{error_id}] CUDA out of memory error. Consider: "
                        "1) reducing image size, 2) clearing GPU cache, "
                        "3) using smaller batch sizes"
                    )
                elif "device" in error_str or "illegal" in error_str:
                    logger.warning(
                        f"[{error_id}] CUDA device error detected. GPU state may be corrupted. "
                        "Service restart may be required."
                    )

            context.abort(
                grpc.StatusCode.INTERNAL,
                f"Prediction failed with error: {repr(e)} (error_id: {error_id})",
            )

    def RunDetectionStream(self, request_iterator, context):
        """
        Handle streaming detection requests.

        Accumulates request data from the stream and calls RunDetection.
        """
        request = proto.DetectionRequest()

        for next_request in request_iterator:
            if next_request.image_data.HasField("pixels"):
                request.image_data.pixels.CopyFrom(next_request.image_data.pixels)

            if next_request.image_data.HasField("image_annotation"):
                request.image_data.image_annotation.CopyFrom(
                    next_request.image_data.image_annotation
                )

            if next_request.HasField("detection_settings"):
                request.detection_settings.CopyFrom(next_request.detection_settings)

            if request.image_data.HasField("pixels"):
                yield self.RunDetection(request, context)