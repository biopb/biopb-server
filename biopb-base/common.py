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


def _validate_type(key: str, value, expected_type: str, spec: dict) -> str | None:
    """
    Validate the type of a single value.

    Args:
        key: Parameter name (for error messages)
        value: The value to validate
        expected_type: Expected type string ("int", "number", "bool", "string", "array")
        spec: Full spec dict (used for array item_type validation)

    Returns:
        Error message string if invalid, None if valid
    """
    if expected_type == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            return f"Parameter '{key}' must be an integer, got {type(value).__name__}"

    elif expected_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"Parameter '{key}' must be a number, got {type(value).__name__}"

    elif expected_type == "bool":
        if not isinstance(value, bool):
            return f"Parameter '{key}' must be a boolean, got {type(value).__name__}"

    elif expected_type == "string":
        if not isinstance(value, str):
            return f"Parameter '{key}' must be a string, got {type(value).__name__}"

    elif expected_type == "array":
        if not isinstance(value, list):
            return f"Parameter '{key}' must be an array, got {type(value).__name__}"

        # Validate array items
        item_type = spec.get("item_type")
        if item_type:
            for i, item in enumerate(value):
                item_error = _validate_type(f"{key}[{i}]", item, item_type, {})
                if item_error:
                    return item_error

    return None


def validate_kwargs(kwargs: dict, schema: dict) -> list[str]:
    """
    Validate kwargs against a schema.

    Args:
        kwargs: Dictionary of parameter values to validate
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
    """
    errors = []

    # Check for unknown parameters
    for key in kwargs:
        if key not in schema:
            valid_params = list(schema.keys())
            errors.append(f"Unknown parameter '{key}'. Valid parameters: {valid_params}")

    # Check each known parameter
    for key, spec in schema.items():
        if key not in kwargs:
            continue

        value = kwargs[key]
        expected_type = spec.get("type")

        # Type validation
        type_error = _validate_type(key, value, expected_type, spec)
        if type_error:
            errors.append(type_error)
            continue  # Skip range validation if type is wrong

        # Range validation for numbers
        if expected_type in ("number", "int"):
            if "minimum" in spec and value < spec["minimum"]:
                errors.append(f"Parameter '{key}' value {value} is below minimum {spec['minimum']}")
            if "maximum" in spec and value > spec["maximum"]:
                errors.append(f"Parameter '{key}' value {value} exceeds maximum {spec['maximum']}")

        # Length validation for arrays
        if expected_type == "array":
            if "min_length" in spec and len(value) < spec["min_length"]:
                errors.append(f"Parameter '{key}' must have at least {spec['min_length']} items")
            if "max_length" in spec and len(value) > spec["max_length"]:
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
    - Thread-safe request handling via lock
    - Error handling with proper gRPC status codes
    - Request logging context

    Subclasses should implement RunDetection and Run methods.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._debug = False

    def set_debug(self, debug: bool):
        """Enable or disable debug mode."""
        self._debug = debug

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
            with self._lock:
                yield

        except ValueError as e:
            logger.error(f"Invalid argument: {e}")
            if self._debug:
                logger.error(traceback.format_exc())
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, repr(e))

        except NotImplementedError as e:
            logger.error(f"Not implemented: {e}")
            context.abort(grpc.StatusCode.UNIMPLEMENTED, repr(e))

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            logger.error(traceback.format_exc())
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"Prediction failed with error: {repr(e)}",
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