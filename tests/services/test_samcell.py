"""Tests for Samcell service."""

import pytest
import numpy as np

import biopb.image as proto
from biopb.image.utils import serialize_from_numpy_to_image_data
from tests.test_service_base import ServiceTestBase


class TestSamcellSmoke(ServiceTestBase):
    """Smoke tests for Samcell service."""

    service_fixture_name = "samcell_service"

    @pytest.mark.skip(reason="Samcell does not implement GetOpNames")
    def test_get_op_names(self, request):
        pass


class TestSamcellIntegration:
    """Integration tests for Samcell-specific features."""

    @pytest.mark.integration
    def test_2d_detection(self, request, test_image_2d):
        """2D cell segmentation should work."""
        service = request.getfixturevalue("samcell_service")
        stub = service.detection_stub()

        image_data = serialize_from_numpy_to_image_data(test_image_2d)
        request_msg = proto.DetectionRequest(
            image_data=image_data,
            detection_settings=proto.DetectionSettings(scaling_hint=1.0),
        )

        response = stub.RunDetection(request_msg, timeout=120)  # Samcell may be slower
        assert len(response.detections) > 0

    @pytest.mark.integration
    def test_2d_process_mask(self, request, test_image_2d):
        """ProcessImage should return segmentation mask."""
        service = request.getfixturevalue("samcell_service")
        stub = service.process_stub()

        image_data = serialize_from_numpy_to_image_data(test_image_2d)
        request_msg = proto.ProcessRequest(image_data=image_data)

        response = stub.Run(request_msg, timeout=120)

        from biopb.image.utils import deserialize_image_data
        result = deserialize_image_data(response.image_data)
        assert result.shape[:2] == test_image_2d.shape[:2]

    