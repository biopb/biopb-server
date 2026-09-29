"""Tests for the unifmir (UNiFMIR restoration) service.

An Ops server with one op per task head. They need a biopb SDK with the Ops
protocol, and the unifmir:test image.
"""

import os
import sys

import numpy as np
import pytest

import grpc
import biopb.image as proto
from biopb.image import deserialize_image_data, serialize_from_numpy_to_image_data
from google.protobuf.empty_pb2 import Empty
from grpc_health.v1 import health_pb2, health_pb2_grpc

_UNIFMIR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "unifmir")


# --------------------------------------------------------------------------- #
# Unit test: vendored model package runs without a checkpoint or Docker.
# --------------------------------------------------------------------------- #
@pytest.mark.smoke
def test_vendored_swinir_forward():
    """The vendored SwinIR backbone instantiates and 2x-upscales a 2D tensor."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("einops")

    sys.path.insert(0, os.path.abspath(_UNIFMIR_DIR))
    try:
        from model.swinir import swinir as SwinIR
    finally:
        sys.path.pop(0)

    model = SwinIR(upscale=2, in_chans=1).eval()
    x = torch.zeros(1, 1, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (1, 1, 128, 128)


# --------------------------------------------------------------------------- #
# Service tests (require a pre-built unifmir:test image).
# --------------------------------------------------------------------------- #
_HEADS = {
    "sr_factin": "YX", "sr_ccps": "YX", "sr_er": "YX", "sr_microtubules": "YX",
    "denoise_planaria": "ZYX", "denoise_tribolium": "ZYX", "isotropic_liver": "ZYX",
}


def _call(stub, op, image, dim_labels, timeout=300):
    """The op's output for *image*, and the result's axes."""
    args = {
        "image": proto.Arg(
            eager=serialize_from_numpy_to_image_data(image, dim_labels=dim_labels).eager_data
        )
    }
    events = list(stub.Call(proto.Call(op=op, args=args), timeout=timeout))
    result = events[-1].outputs["result"].eager
    return deserialize_image_data(proto.ImageData(eager_data=result)), list(result.dim_labels)


class TestUnifmirService:
    @pytest.mark.smoke
    def test_health_check(self, unifmir_channel):
        stub = health_pb2_grpc.HealthStub(unifmir_channel)
        response = stub.Check(health_pb2.HealthCheckRequest(), timeout=5)
        assert response.status == health_pb2.HealthCheckResponse.SERVING

    @pytest.mark.smoke
    def test_describe_lists_every_head(self, unifmir_ops_stub):
        infos = {i.name: i for i in unifmir_ops_stub.Describe(Empty(), timeout=10).ops}
        assert set(infos) == set(_HEADS)
        for name, axes in _HEADS.items():
            assert infos[name].tensors["image"].axes == axes
            assert infos[name].input == proto.OpInfo.LAZY
            assert infos[name].description

    @pytest.mark.integration
    def test_sr_doubles_the_plane(self, unifmir_ops_stub):
        image = (np.random.default_rng(0).random((64, 64)) * 255).astype(np.float32)
        result, labels = _call(unifmir_ops_stub, "sr_factin", image, ["Y", "X"])
        assert labels == ["Y", "X"]
        assert result.shape == (128, 128)
        assert np.isfinite(np.asarray(result)).all()

    @pytest.mark.integration
    def test_denoise_keeps_the_stack_shape(self, unifmir_ops_stub):
        stack = (np.random.default_rng(1).random((4, 64, 64)) * 1000).astype(np.float32)
        result, labels = _call(unifmir_ops_stub, "denoise_planaria", stack, ["Z", "Y", "X"])
        assert labels == ["Z", "Y", "X"]
        assert result.shape == stack.shape
        assert np.isfinite(np.asarray(result)).all()

    @pytest.mark.integration
    def test_stack_to_a_2d_op_is_invalid_argument(self, unifmir_ops_stub):
        stack = np.zeros((3, 32, 32), dtype=np.float32)
        with pytest.raises(grpc.RpcError) as exc:
            _call(unifmir_ops_stub, "sr_factin", stack, ["Z", "Y", "X"])
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    @pytest.mark.integration
    def test_unknown_op_is_not_found(self, unifmir_ops_stub):
        image = np.zeros((32, 32), dtype=np.float32)
        with pytest.raises(grpc.RpcError) as exc:
            _call(unifmir_ops_stub, "does_not_exist", image, ["Y", "X"])
        assert exc.value.code() == grpc.StatusCode.NOT_FOUND
