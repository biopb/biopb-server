"""Tests for the samcell service, an Ops server.

They need a biopb SDK with the Ops protocol, and the samcell:test image.
"""

import grpc
import numpy as np
import pytest
from google.protobuf import empty_pb2
from grpc_health.v1 import health_pb2, health_pb2_grpc

import biopb.image as proto
from biopb.image import deserialize_image_data, serialize_from_numpy_to_image_data


def _call(stub, image, dim_labels):
    """The samcell op's label image for *image*, and the result's axes."""
    args = {
        "image": proto.Arg(
            eager=serialize_from_numpy_to_image_data(image, dim_labels=dim_labels).eager_data
        )
    }
    events = list(stub.Call(proto.Call(op="samcell", args=args), timeout=300))
    result = events[-1].outputs["result"].eager
    return deserialize_image_data(proto.ImageData(eager_data=result)), list(result.dim_labels)


class TestSamcellSmoke:
    """Smoke tests for the samcell service."""

    @pytest.mark.smoke
    def test_health_check(self, samcell_channel):
        stub = health_pb2_grpc.HealthStub(samcell_channel)
        response = stub.Check(health_pb2.HealthCheckRequest(), timeout=5)
        assert response.status == health_pb2.HealthCheckResponse.SERVING

    @pytest.mark.smoke
    def test_describe_lists_samcell(self, samcell_ops_stub):
        (info,) = samcell_ops_stub.Describe(empty_pb2.Empty(), timeout=10).ops
        assert info.name == "samcell"
        assert list(info.tensors) == ["image"]
        assert info.input == proto.OpInfo.EAGER


class TestSamcellIntegration:
    """Integration tests for the samcell op."""

    @pytest.mark.integration
    def test_2d_label_image(self, samcell_ops_stub, test_image_2d):
        mask, labels = _call(samcell_ops_stub, test_image_2d, ["Y", "X"])
        assert labels == ["Y", "X"]
        assert mask.shape == test_image_2d.shape
        assert mask.max() > 0

    @pytest.mark.integration
    def test_multichannel_is_averaged(self, samcell_ops_stub, test_image_2d):
        rgb = np.stack([test_image_2d] * 3, -1)
        mask, labels = _call(samcell_ops_stub, rgb, ["Y", "X", "C"])
        assert labels == ["Y", "X", "C"]
        assert mask.shape[:2] == test_image_2d.shape

    @pytest.mark.integration
    def test_3d_is_invalid_argument(self, samcell_ops_stub, test_image_2d):
        volume = np.stack([test_image_2d[:128, :128]] * 3)
        with pytest.raises(grpc.RpcError) as info:
            _call(samcell_ops_stub, volume, ["Z", "Y", "X"])
        assert info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
