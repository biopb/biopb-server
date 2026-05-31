"""Tests for the unifmr (UNiFMIR restoration) service.

UNiFMIR is a restoration model (image -> image), so the suite is built around
``ProcessImage.Run`` rather than the detection-centric ``ServiceTestBase``.
``RunDetection`` is expected to be unimplemented.
"""

import os
import sys

import numpy as np
import pytest

import grpc
import biopb.image as proto
from biopb.image.utils import serialize_from_numpy_to_image_data, deserialize_image_data
from grpc_health.v1 import health_pb2, health_pb2_grpc

_UNIFMR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "unifmr")


# --------------------------------------------------------------------------- #
# Unit test: vendored model package runs without a checkpoint or Docker.
# --------------------------------------------------------------------------- #
@pytest.mark.smoke
def test_vendored_swinir_forward():
    """The vendored SwinIR backbone instantiates and 2x-upscales a 2D tensor."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("einops")

    sys.path.insert(0, os.path.abspath(_UNIFMR_DIR))
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
# Service tests (require a pre-built unifmr:test image).
# --------------------------------------------------------------------------- #
class TestUnifmrService:
    @pytest.mark.smoke
    def test_health_check(self, unifmr_service):
        stub = health_pb2_grpc.HealthStub(unifmr_service.channel())
        response = stub.Check(health_pb2.HealthCheckRequest(), timeout=5)
        assert response.status == health_pb2.HealthCheckResponse.SERVING

    @pytest.mark.smoke
    def test_get_op_names(self, unifmr_service):
        from google.protobuf.empty_pb2 import Empty

        stub = unifmr_service.process_stub()
        response = stub.GetOpNames(Empty(), timeout=10)
        assert "sr_factin" in response.names
        assert "denoise_planaria" in response.names
        assert "isotropic_liver" in response.names

    @pytest.mark.integration
    def test_sr_process_roundtrip(self, unifmr_service):
        """SR (sr_factin) returns a finite 2x-upscaled 2D image."""
        stub = unifmr_service.process_stub()
        image = (np.random.default_rng(0).random((64, 64)) * 255).astype(np.float32)

        request_msg = proto.ProcessRequest(
            image_data=serialize_from_numpy_to_image_data(image),
            op_name="sr_factin",
        )
        response = stub.Run(request_msg, timeout=120)

        result = deserialize_image_data(response.image_data)
        assert result.shape == (128, 128)
        assert np.isfinite(np.asarray(result)).all()

    @pytest.mark.integration
    def test_unknown_op_rejected(self, unifmr_service):
        """An unknown op_name yields INVALID_ARGUMENT."""
        stub = unifmr_service.process_stub()
        image = np.zeros((32, 32), dtype=np.float32)

        request_msg = proto.ProcessRequest(
            image_data=serialize_from_numpy_to_image_data(image),
            op_name="does_not_exist",
        )
        with pytest.raises(grpc.RpcError) as exc:
            stub.Run(request_msg, timeout=30)
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    @pytest.mark.integration
    def test_run_detection_unimplemented(self, unifmr_service):
        """RunDetection is not supported for a restoration model."""
        stub = unifmr_service.detection_stub()
        image = np.zeros((32, 32), dtype=np.float32)

        request_msg = proto.DetectionRequest(
            image_data=serialize_from_numpy_to_image_data(image),
        )
        with pytest.raises(grpc.RpcError) as exc:
            stub.RunDetection(request_msg, timeout=30)
        assert exc.value.code() == grpc.StatusCode.UNIMPLEMENTED
