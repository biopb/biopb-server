"""Fast, GPU-free unit tests for the shared biopb_image_base utilities.

Every *_server.py builds its request handling on these helpers (kwargs merge,
schema validation, lazy-data rejection, image codec). These tests exercise that
shared contract directly -- no Docker container, no model, no GPU -- so they run
in milliseconds against the `biopb-image-base` package installed in the venv.

Service-specific logic (process_input/process_result, per-service schemas) still
needs the integration tests in tests/services/ because importing a *_server.py
pulls in its ML stack (torch/jax/cellpose).
"""

import dask.array as da
import numpy as np
import pytest
from google.protobuf.struct_pb2 import Struct

import biopb.image as proto
from biopb_image_base import (
    decode_image_data,
    encode_image,
    ensure_eager,
    parse_kwargs,
    validate_kwargs,
)

pytestmark = pytest.mark.unit


# A representative schema mirroring the shape services declare (e.g. cellpose's
# channels/diameter/normalize). Kept local so these tests don't depend on
# importing a service module.
_SCHEMA = {
    "channels": {"type": "array", "item_type": "int", "min_length": 2, "max_length": 2},
    "diameter": {"type": "number", "minimum": 0},
    "min_size": {"type": "int", "minimum": 0},
    "normalize": {"type": "bool"},
    "model_type": {"type": "string"},
}


def _request_with_kwargs(values: dict) -> proto.DetectionRequest:
    """Build a DetectionRequest whose kwargs Struct holds `values`."""
    s = Struct()
    s.update(values)
    img = encode_image(np.zeros((1, 8, 8, 1), dtype="uint8"))
    return proto.DetectionRequest(image_data=img, kwargs=s)


# --------------------------------------------------------------------------
# parse_kwargs
# --------------------------------------------------------------------------

def test_parse_kwargs_no_kwargs_returns_defaults_copy():
    defaults = {"diameter": 30.0}
    req = proto.DetectionRequest()  # no kwargs field set
    assert not req.HasField("kwargs")

    out = parse_kwargs(req, defaults)
    assert out == {"diameter": 30.0}
    # Must be a copy -- mutating the result must not touch the caller's defaults.
    out["diameter"] = 99.0
    assert defaults["diameter"] == 30.0


def test_parse_kwargs_overrides_and_adds_keys():
    req = _request_with_kwargs({"diameter": 50.0, "min_size": 7})
    out = parse_kwargs(req, {"diameter": 30.0, "normalize": True})
    assert out["diameter"] == 50.0      # overridden
    assert out["normalize"] is True     # untouched default
    assert out["min_size"] == 7         # added from request


def test_parse_kwargs_converts_protobuf_value_types():
    req = _request_with_kwargs(
        {"diameter": 12.5, "normalize": True, "model_type": "cyto3", "channels": [1, 2]}
    )
    out = parse_kwargs(req, {})
    assert out["diameter"] == 12.5
    assert out["normalize"] is True
    assert out["model_type"] == "cyto3"
    # protobuf has no int scalar -> numbers come back as float
    assert out["channels"] == [1.0, 2.0]


# --------------------------------------------------------------------------
# validate_kwargs
# --------------------------------------------------------------------------

def test_validate_kwargs_accepts_valid():
    kwargs = {"channels": [1, 2], "diameter": 30.0, "min_size": 15, "normalize": True}
    assert validate_kwargs(kwargs, _SCHEMA) == []


def test_validate_kwargs_ignores_unknown_keys():
    # Keys absent from the schema are not validated (and not errors).
    assert validate_kwargs({"not_in_schema": object()}, _SCHEMA) == []


def test_validate_kwargs_accepts_int_valued_float():
    # protobuf delivers ints as floats; an integer-valued float must pass "int".
    assert validate_kwargs({"min_size": 15.0, "channels": [1.0, 2.0]}, _SCHEMA) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"diameter": -1.0},          # below minimum
        {"min_size": -5},            # below minimum
        {"normalize": "yes"},        # wrong type
        {"channels": [1]},           # too short
        {"channels": [1, 2, 3]},     # too long
        {"diameter": "big"},         # wrong type
    ],
)
def test_validate_kwargs_reports_errors(kwargs):
    errors = validate_kwargs(kwargs, _SCHEMA)
    assert len(errors) >= 1
    # error message is keyed by the offending field
    assert any(next(iter(kwargs)) in e for e in errors)


# --------------------------------------------------------------------------
# ensure_eager
# --------------------------------------------------------------------------

def test_ensure_eager_passes_numpy_through():
    arr = np.ones((4, 4), dtype="uint8")
    assert ensure_eager(arr) is arr


def test_ensure_eager_rejects_dask():
    lazy = da.from_array(np.zeros((4, 4)), chunks=2)
    with pytest.raises(ValueError):
        ensure_eager(lazy)


# --------------------------------------------------------------------------
# image codec round-trip
# --------------------------------------------------------------------------

def test_encode_decode_roundtrip_preserves_shape_and_values():
    rng = np.random.default_rng(0)
    arr = (rng.random((1, 32, 32, 1)) * 255).astype("uint8")

    decoded = decode_image_data(encode_image(arr))

    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == arr.shape
    np.testing.assert_array_equal(decoded, arr)
