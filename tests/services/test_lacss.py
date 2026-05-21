"""Tests for Lacss service."""

import pytest
import numpy as np

import biopb.image as proto
from biopb.image.utils import serialize_from_numpy_to_image_data
from tests.test_service_base import ServiceTestBase


class TestLacssSmoke(ServiceTestBase):
    """Smoke tests for Lacss service."""

    service_fixture_name = "lacss_service"

    @pytest.mark.skip(reason="Lacss does not implement GetOpNames")
    def test_get_op_names(self, request):
        pass


class TestLacssIntegration:
    """Integration tests for Lacss-specific features."""

    @pytest.mark.integration
    def test_2d_detection(self, request, test_image_2d):
        """2D cell segmentation should work."""
        service = request.getfixturevalue("lacss_service")
        stub = service.detection_stub()

        image_data = serialize_from_numpy_to_image_data(test_image_2d)
        request_msg = proto.DetectionRequest(
            image_data=image_data,
            detection_settings=proto.DetectionSettings(scaling_hint=1.0),
        )

        response = stub.RunDetection(request_msg, timeout=60)
        assert len(response.detections) > 0

    @pytest.mark.integration
    def test_detection_with_scores(self, request, test_image_2d):
        """Lacss should return detection scores."""
        service = request.getfixturevalue("lacss_service")
        stub = service.detection_stub()

        image_data = serialize_from_numpy_to_image_data(test_image_2d)
        request_msg = proto.DetectionRequest(image_data=image_data)

        response = stub.RunDetection(request_msg, timeout=60)

        # Lacss returns confidence scores
        for det in response.detections:
            assert det.score >= 0.0

    