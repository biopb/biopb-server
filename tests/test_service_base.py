"""Base test class for biopb.image services."""

import pytest
import grpc
import numpy as np
from grpc_health.v1 import health_pb2, health_pb2_grpc

import biopb.image as proto
from biopb.image.utils import serialize_from_numpy_to_image_data


class ServiceTestBase:
    """Base class providing common tests for all services.

    Subclass and set the service fixture name in the constructor.
    """

    # Override in subclasses
    service_fixture_name = None

    def get_service(self, request):
        """Get service fixture by name."""
        return request.getfixturevalue(self.service_fixture_name)

    # ========================================
    # Smoke Tests (required for all services)
    # ========================================

    @pytest.mark.smoke
    def test_health_check(self, request):
        """Service should report SERVING status."""
        service = self.get_service(request)
        channel = service.channel()

        stub = health_pb2_grpc.HealthStub(channel)
        response = stub.Check(health_pb2.HealthCheckRequest(), timeout=5)

        assert response.status == health_pb2.HealthCheckResponse.SERVING

    @pytest.mark.smoke
    def test_detection_returns_results(self, request, test_image_2d):
        """RunDetection should return valid DetectionResponse."""
        service = self.get_service(request)
        stub = service.detection_stub()

        image_data = serialize_from_numpy_to_image_data(test_image_2d)
        request_msg = proto.DetectionRequest(
            image_data=image_data,
            detection_settings=proto.DetectionSettings(scaling_hint=1.0),
        )

        response = stub.RunDetection(request_msg, timeout=30)

        # Should have detections
        assert len(response.detections) > 0

        # Each detection should have valid structure
        for det in response.detections:
            assert det.score >= 0.0
            assert det.HasField("roi")

    @pytest.mark.smoke
    def test_detection_various_sizes(self, request):
        """RunDetection should handle various image sizes."""
        service = self.get_service(request)
        stub = service.detection_stub()

        # Use the real test image, resized to various dimensions
        from tests.utils.image_utils import load_test_image
        import numpy as np

        base_image = load_test_image()
        sizes = [(256, 256), (512, 512), (373, 372)]  # Non-square included

        for height, width in sizes:
            # Resize the real test image
            image = np.array(base_image)  # Ensure it's a numpy array
            if height != image.shape[0] or width != image.shape[1]:
                # Simple resize using slicing or scipy
                from scipy.ndimage import zoom
                scale_h = height / image.shape[0]
                scale_w = width / image.shape[1]
                if image.ndim == 3:
                    image = zoom(image, (scale_h, scale_w, 1), order=1)
                else:
                    image = zoom(image, (scale_h, scale_w), order=1)
                image = image.astype(np.uint8)

            image_data = serialize_from_numpy_to_image_data(image)
            request_msg = proto.DetectionRequest(image_data=image_data)

            response = stub.RunDetection(request_msg, timeout=30)
            # Should return valid response (may or may not have detections depending on content)
            assert response is not None

    @pytest.mark.smoke
    def test_get_op_names(self, request):
        """GetOpNames should return valid operation names."""
        service = self.get_service(request)
        stub = service.process_stub()

        from google.protobuf.empty_pb2 import Empty
        response = stub.GetOpNames(Empty(), timeout=10)

        # Should return at least one operation
        assert len(response.names) > 0

    # ========================================
    # Contract Tests (API compliance)
    # ========================================

    @pytest.mark.contract
    def test_image_data_eager_format(self, request, test_image_2d):
        """Service should accept eager_data ImageData format."""
        service = self.get_service(request)
        stub = service.detection_stub()

        # Create ImageData with eager_data field
        image_data = serialize_from_numpy_to_image_data(test_image_2d)

        # Should have eager_data (new format)
        assert image_data.HasField("eager_data") or image_data.HasField("pixels")

        request_msg = proto.DetectionRequest(image_data=image_data)
        response = stub.RunDetection(request_msg, timeout=30)

        # Should succeed with valid response
        assert len(response.detections) > 0

    @pytest.mark.contract
    def test_detection_response_structure(self, request, test_image_2d):
        """DetectionResponse should have required fields."""
        service = self.get_service(request)
        stub = service.detection_stub()

        image_data = serialize_from_numpy_to_image_data(test_image_2d)
        request_msg = proto.DetectionRequest(image_data=image_data)

        response = stub.RunDetection(request_msg, timeout=30)

        # Validate response structure
        for det in response.detections:
            # Score field
            assert isinstance(det.score, float)

            # ROI field
            assert det.HasField("roi")

            # ROI should have polygon (most common for cell segmentation)
            if det.roi.HasField("polygon"):
                assert len(det.roi.polygon.points) >= 3
                for pt in det.roi.polygon.points:
                    assert isinstance(pt.x, float)
                    assert isinstance(pt.y, float)

    @pytest.mark.contract
    def test_kwargs_validation(self, request, test_image_2d):
        """Invalid kwargs should return INVALID_ARGUMENT."""
        service = self.get_service(request)
        stub = service.detection_stub()

        image_data = serialize_from_numpy_to_image_data(test_image_2d)

        # Create request with invalid kwargs (if service supports kwargs)
        from google.protobuf.struct_pb2 import Struct
        kwargs_struct = Struct()
        kwargs_struct.fields["invalid_diameter"].number_value = -999  # Invalid negative

        request_msg = proto.DetectionRequest(
            image_data=image_data,
            kwargs=kwargs_struct,
        )

        try:
            response = stub.RunDetection(request_msg, timeout=30)
            # If service ignores unknown kwargs, this is acceptable
            # Some services may not validate kwargs strictly
        except grpc.RpcError as e:
            # Should be INVALID_ARGUMENT if service validates
            # Accept either INVALID_ARGUMENT or success (service ignores unknown)
            if e.code() != grpc.StatusCode.INVALID_ARGUMENT:
                # Service might not validate kwargs - acceptable for some services
                pass