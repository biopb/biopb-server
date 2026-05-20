"""Shared utilities for biopb-server services.

This module provides utilities not available in biopb_image_base that are
needed by some services (e.g., kwargs validation).
"""

import logging
from typing import Any

from google.protobuf.json_format import MessageToDict

logger = logging.getLogger(__name__)


def parse_kwargs(request, defaults: dict) -> dict:
    """Merge request kwargs with defaults.

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
    """Attempt to coerce a value to the expected type.

    Returns (coerced_value, warning_message) where warning_message is None on success.
    """
    warning = None

    if expected_type == "int":
        if isinstance(value, bool):
            warning = "Parameter is a boolean, expected integer"
        elif isinstance(value, int):
            pass
        elif isinstance(value, float):
            coerced = int(value)
            if coerced != value:
                warning = f"Parameter value {value} truncated to {coerced} for integer type"
            return coerced, warning
        else:
            warning = f"Parameter type {type(value).__name__} cannot be converted to integer"

    elif expected_type == "number":
        if isinstance(value, bool):
            warning = "Parameter is a boolean, expected number"
        elif isinstance(value, (int, float)):
            pass
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
            return list(value), warning
        elif not isinstance(value, list):
            warning = f"Parameter type {type(value).__name__}, expected array"

    return value, warning


def _validate_type(key: str, value, expected_type: str, spec: dict) -> tuple[Any, str | None]:
    """Validate and optionally coerce a value to the expected type."""
    coerced_value, warning = _coerce_value(value, expected_type)

    if warning:
        logger.warning(f"Parameter '{key}': {warning}")

    if expected_type == "array" and isinstance(coerced_value, list):
        item_type = spec.get("item_type")
        if item_type:
            coerced_items = []
            for i, item in enumerate(coerced_value):
                item_value, item_warning = _coerce_value(item, item_type)
                if item_warning:
                    logger.warning(f"Parameter '{key}[{i}]': {item_warning}")
                coerced_items.append(item_value)
            coerced_value = coerced_items

    return coerced_value, None


def validate_kwargs(kwargs: dict, schema: dict) -> list[str]:
    """Validate and coerce kwargs against a schema.

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
                "description": str (optional),
            },
        }
    """
    errors = []

    for key in kwargs:
        if key not in schema:
            logger.warning(f"Unknown parameter '{key}' will be ignored")

    for key, spec in schema.items():
        if key not in kwargs:
            continue

        value = kwargs[key]
        expected_type = spec.get("type")

        coerced_value, type_error = _validate_type(key, value, expected_type, spec)
        if type_error:
            errors.append(type_error)
            continue

        kwargs[key] = coerced_value

        if expected_type in ("number", "int"):
            if "minimum" in spec and coerced_value < spec["minimum"]:
                errors.append(f"Parameter '{key}' value {coerced_value} is below minimum {spec['minimum']}")
            if "maximum" in spec and coerced_value > spec["maximum"]:
                errors.append(f"Parameter '{key}' value {coerced_value} exceeds maximum {spec['maximum']}")

        if expected_type == "array":
            if "min_length" in spec and len(coerced_value) < spec["min_length"]:
                errors.append(f"Parameter '{key}' must have at least {spec['min_length']} items")
            if "max_length" in spec and len(coerced_value) > spec["max_length"]:
                errors.append(f"Parameter '{key}' must have at most {spec['max_length']} items")

    return errors