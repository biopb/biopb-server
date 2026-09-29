"""Tests for the cellpose-sam service, an Ops server.

They need a biopb SDK with the Ops protocol, and the cellpose-sam:test image.
"""

import grpc
import numpy as np
import pytest
from google.protobuf import empty_pb2, json_format, struct_pb2
from grpc_health.v1 import health_pb2, health_pb2_grpc

import biopb.image as proto
from biopb.image import deserialize_image_data, serialize_from_numpy_to_image_data


def _call(stub, image, dim_labels, timeout=180, **kwargs):
    """The cellpose op's label image for *image*, and the result's axes."""
    args = {
        "image": proto.Arg(
            eager=serialize_from_numpy_to_image_data(image, dim_labels=dim_labels).eager_data
        )
    }
    for key, value in kwargs.items():
        args[key] = proto.Arg(json=json_format.ParseDict(value, struct_pb2.Value()))
    events = list(stub.Call(proto.Call(op="cellpose", args=args), timeout=timeout))
    result = events[-1].outputs["result"].eager
    return deserialize_image_data(proto.ImageData(eager_data=result)), list(result.dim_labels)


class TestCellposeSamSmoke:
    """Smoke tests for the cellpose-sam service."""

    @pytest.mark.smoke
    def test_health_check(self, cellpose_sam_channel):
        stub = health_pb2_grpc.HealthStub(cellpose_sam_channel)
        response = stub.Check(health_pb2.HealthCheckRequest(), timeout=5)
        assert response.status == health_pb2.HealthCheckResponse.SERVING

    @pytest.mark.smoke
    def test_describe_lists_cellpose(self, cellpose_sam_ops_stub):
        (info,) = cellpose_sam_ops_stub.Describe(empty_pb2.Empty(), timeout=10).ops
        assert info.name == "cellpose"
        assert list(info.tensors) == ["image"]
        assert "cellprob_threshold=0.0" in info.kwargs
        assert info.input == proto.OpInfo.LAZY


class TestCellposeSamIntegration:
    """Integration tests for the cellpose-sam op."""

    @pytest.mark.integration
    def test_2d_label_image(self, cellpose_sam_ops_stub, test_image_2d):
        mask, labels = _call(cellpose_sam_ops_stub, test_image_2d, ["Y", "X"])
        assert labels == ["Y", "X"]
        assert mask.shape == test_image_2d.shape
        assert mask.max() > 0

    @pytest.mark.integration
    def test_diameter_kwarg(self, cellpose_sam_ops_stub, test_image_2d):
        mask, _ = _call(cellpose_sam_ops_stub, test_image_2d, ["Y", "X"], diameter=30.0)
        assert mask.max() > 0

    @pytest.mark.integration
    def test_invalid_kwarg_is_invalid_argument(self, cellpose_sam_ops_stub, test_image_2d):
        with pytest.raises(grpc.RpcError) as info:
            _call(cellpose_sam_ops_stub, test_image_2d, ["Y", "X"], cellprob_threshold=99.0)
        assert info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    @pytest.mark.integration
    def test_multichannel(self, cellpose_sam_ops_stub, test_image_2d):
        rgb = np.stack([test_image_2d, test_image_2d // 2, np.zeros_like(test_image_2d)], -1)
        mask, labels = _call(cellpose_sam_ops_stub, rgb, ["Y", "X", "C"])
        assert labels == ["Y", "X", "C"]
        assert mask.shape[:2] == test_image_2d.shape
        assert mask.max() > 0

    @pytest.mark.integration
    def test_3d_volume(self, cellpose_sam_ops_stub, test_image_2d):
        volume = np.stack([test_image_2d[:128, :128]] * 4)  # 3D runs the net on 3 orientations: slow
        mask, labels = _call(cellpose_sam_ops_stub, volume, ["Z", "Y", "X"], timeout=900)
        assert labels == ["Z", "Y", "X"]
        assert mask.shape == volume.shape
