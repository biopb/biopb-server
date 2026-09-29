"""Tests for the ucell service, an Ops server.

They need a biopb SDK with the Ops protocol, and the ucell:test image.
"""

import grpc
import numpy as np
import pytest
from google.protobuf import empty_pb2, json_format, struct_pb2
from grpc_health.v1 import health_pb2, health_pb2_grpc

import biopb.image as proto
from biopb.image import deserialize_image_data, serialize_from_numpy_to_image_data


def _call(stub, image, dim_labels, **kwargs):
    """The ucell op's label image for *image*, and the result's axes."""
    args = {
        "image": proto.Arg(
            eager=serialize_from_numpy_to_image_data(image, dim_labels=dim_labels).eager_data
        )
    }
    for key, value in kwargs.items():
        args[key] = proto.Arg(json=json_format.ParseDict(value, struct_pb2.Value()))
    events = list(stub.Call(proto.Call(op="ucell", args=args), timeout=120))
    result = events[-1].outputs["result"].eager
    return deserialize_image_data(proto.ImageData(eager_data=result)), list(result.dim_labels)


class TestUcellSmoke:
    """Smoke tests for the ucell service."""

    @pytest.mark.smoke
    def test_health_check(self, ucell_channel):
        stub = health_pb2_grpc.HealthStub(ucell_channel)
        response = stub.Check(health_pb2.HealthCheckRequest(), timeout=5)
        assert response.status == health_pb2.HealthCheckResponse.SERVING

    @pytest.mark.smoke
    def test_describe_lists_ucell(self, ucell_ops_stub):
        (info,) = ucell_ops_stub.Describe(empty_pb2.Empty(), timeout=10).ops
        assert info.name == "ucell"
        assert list(info.tensors) == ["image"]
        assert "min_area=5" in info.kwargs
        assert info.input == proto.OpInfo.LAZY


class TestUcellIntegration:
    """Integration tests for the ucell op."""

    @pytest.mark.integration
    def test_2d_label_image(self, ucell_ops_stub, test_image_2d):
        mask, labels = _call(ucell_ops_stub, test_image_2d, ["Y", "X"])
        assert labels == ["Y", "X"]
        assert mask.shape == test_image_2d.shape
        assert mask.max() > 0

    @pytest.mark.integration
    def test_kwargs(self, ucell_ops_stub, test_image_2d):
        mask, _ = _call(
            ucell_ops_stub, test_image_2d, ["Y", "X"],
            task_id=0, cellprob_threshold=-0.5, min_area=10,
        )
        assert mask.max() > 0

    @pytest.mark.integration
    def test_invalid_kwarg_is_invalid_argument(self, ucell_ops_stub, test_image_2d):
        with pytest.raises(grpc.RpcError) as info:
            _call(ucell_ops_stub, test_image_2d, ["Y", "X"], cellprob_threshold=10.0)
        assert info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    @pytest.mark.integration
    def test_channel_first_input(self, ucell_ops_stub, test_image_2d):
        cyx = np.stack([test_image_2d, test_image_2d // 2])
        mask, labels = _call(ucell_ops_stub, cyx, ["C", "Y", "X"])
        assert labels == ["C", "Y", "X"]
        assert mask.shape[-2:] == test_image_2d.shape

    @pytest.mark.integration
    def test_too_many_channels_is_invalid_argument(self, ucell_ops_stub, test_image_2d):
        yxc = np.stack([test_image_2d] * 4, -1)
        with pytest.raises(grpc.RpcError) as info:
            _call(ucell_ops_stub, yxc, ["Y", "X", "C"])
        assert info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
