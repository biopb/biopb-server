"""Tests for Cellpose service."""

import pytest
import numpy as np

import biopb.image as proto
from biopb.image.utils import serialize_from_numpy_to_image_data
from tests.test_service_base import ServiceTestBase


class TestCellposeSmoke(ServiceTestBase):
    """Smoke tests for Cellpose service."""

    service_fixture_name = "cellpose_service"


class TestCellposeIntegration:
    """Integration tests for Cellpose-specific features."""

    @pytest.mark.integration
    def test_2d_detection(self, cellpose_detection_stub, test_image_2d):
        """2D cell segmentation should work."""
        image_data = serialize_from_numpy_to_image_data(test_image_2d)
        request = proto.DetectionRequest(
            image_data=image_data,
            detection_settings=proto.DetectionSettings(scaling_hint=1.0),
        )

        response = cellpose_detection_stub.RunDetection(request, timeout=30)
        assert len(response.detections) > 0

    @pytest.mark.integration
    def test_2d_process_mask(self, cellpose_process_stub, test_image_2d):
        """ProcessImage should return segmentation mask."""
        image_data = serialize_from_numpy_to_image_data(test_image_2d)
        request = proto.ProcessRequest(image_data=image_data)

        response = cellpose_process_stub.Run(request, timeout=60)

        # Decode result
        from biopb.image.utils import deserialize_image_data
        result = deserialize_image_data(response.image_data)

        # Should be a mask with same spatial dimensions
        assert result.shape[:2] == test_image_2d.shape[:2]

    @pytest.mark.integration
    def test_kwargs_diameter(self, cellpose_detection_stub, test_image_2d):
        """Custom diameter parameter should work."""
        from google.protobuf.struct_pb2 import Struct, Value

        image_data = serialize_from_numpy_to_image_data(test_image_2d)
        kwargs_struct = Struct()
        kwargs_struct.fields["diameter"] = Value(number_value=50.0)

        request = proto.DetectionRequest(
            image_data=image_data,
            detection_settings=proto.DetectionSettings(scaling_hint=1.0),
            kwargs=kwargs_struct,
        )

        response = cellpose_detection_stub.RunDetection(request, timeout=30)
        assert len(response.detections) > 0

    @pytest.mark.integration
    def test_kwargs_channels(self, cellpose_detection_stub, test_image_multichannel):
        """Channel specification should work."""
        from google.protobuf.struct_pb2 import Struct, Value, ListValue

        image_data = serialize_from_numpy_to_image_data(test_image_multichannel)
        kwargs_struct = Struct()
        channel_list = ListValue()
        channel_list.values.append(Value(number_value=1))
        channel_list.values.append(Value(number_value=2))
        kwargs_struct.fields["channels"] = Value(list_value=channel_list)

        request = proto.DetectionRequest(
            image_data=image_data,
            detection_settings=proto.DetectionSettings(scaling_hint=1.0),
            kwargs=kwargs_struct,
        )

        response = cellpose_detection_stub.RunDetection(request, timeout=30)
        assert len(response.detections) > 0

    @pytest.mark.integration
    def test_3d_raises_error(self, cellpose_detection_stub, test_image_3d):
        """3D input to RunDetection should raise appropriate error."""
        # Cellpose RunDetection doesn't support 3D, but ProcessImage does
        image_data = serialize_from_numpy_to_image_data(test_image_3d)
        request = proto.DetectionRequest(
            image_data=image_data,
            detection_settings=proto.DetectionSettings(scaling_hint=1.0),
        )

        with pytest.raises(Exception):  # Should raise ValueError or similar
            cellpose_detection_stub.RunDetection(request, timeout=30)

    @pytest.mark.integration
    def test_get_op_names_returns_cellpose(self, cellpose_process_stub):
        """GetOpNames should return 'cellpose' operation."""
        from google.protobuf.empty_pb2 import Empty

        response = cellpose_process_stub.GetOpNames(Empty(), timeout=10)
        assert "cellpose" in response.names

    @pytest.mark.integration
    def test_cell_diameter_hint(self, cellpose_detection_stub, test_image_2d):
        """cell_diameter_hint should affect detection."""
        image_data = serialize_from_numpy_to_image_data(test_image_2d)

        # Request with diameter hint
        request = proto.DetectionRequest(
            image_data=image_data,
            detection_settings=proto.DetectionSettings(
                scaling_hint=1.0,
                cell_diameter_hint=30.0,  # 30 microns
            ),
        )

        response = cellpose_detection_stub.RunDetection(request, timeout=30)
        assert len(response.detections) > 0