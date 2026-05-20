"""Tests for Cellpose-SAM service."""

import pytest
import numpy as np

import biopb.image as proto
from biopb.image.utils import serialize_from_numpy_to_image_data
from tests.test_service_base import ServiceTestBase


class TestCellposeSamSmoke(ServiceTestBase):
    """Smoke tests for Cellpose-SAM service."""

    service_fixture_name = "cellpose_sam_service"


class TestCellposeSamIntegration:
    """Integration tests for Cellpose-SAM-specific features."""

    @pytest.mark.integration
    def test_2d_detection(self, request, test_image_2d):
        """2D cell segmentation should work."""
        service = request.getfixturevalue("cellpose_sam_service")
        stub = service.detection_stub()

        image_data = serialize_from_numpy_to_image_data(test_image_2d)
        request_msg = proto.DetectionRequest(
            image_data=image_data,
            detection_settings=proto.DetectionSettings(scaling_hint=1.0),
        )

        response = stub.RunDetection(request_msg, timeout=60)
        assert len(response.detections) > 0

    @pytest.mark.integration
    def test_2d_process_mask(self, request, test_image_2d):
        """ProcessImage should return segmentation mask."""
        service = request.getfixturevalue("cellpose_sam_service")
        stub = service.process_stub()

        image_data = serialize_from_numpy_to_image_data(test_image_2d)
        request_msg = proto.ProcessRequest(image_data=image_data)

        response = stub.Run(request_msg, timeout=60)

        from biopb.image.utils import deserialize_image_data
        result = deserialize_image_data(response.image_data)
        assert result.shape[:2] == test_image_2d.shape[:2]

    @pytest.mark.integration
    def test_get_op_names(self, request):
        """GetOpNames should return available operations."""
        service = request.getfixturevalue("cellpose_sam_service")
        stub = service.process_stub()

        from google.protobuf.empty_pb2 import Empty
        response = stub.GetOpNames(Empty(), timeout=10)
        assert len(response.names) > 0