"""Shared pytest fixtures for biopb-server tests.

Note: Most services require GPU. On machines without sufficient GPU,
tests will be skipped. Cellpose is the most lightweight and can often
run on machines with limited GPU memory.
"""

import subprocess
import time
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Optional

import grpc
import numpy as np
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc

import biopb.image as proto
from biopb.image.utils import serialize_from_numpy_to_image_data


# Default gRPC options for large messages
_GRPC_OPTIONS = [
    ("grpc.max_receive_message_length", 256 * 1024 * 1024),  # 256MB
    ("grpc.max_send_message_length", 256 * 1024 * 1024),
]


# Service GPU requirements (approximate VRAM needed)
_SERVICE_GPU_REQUIREMENTS = {
    "cellpose": "1GB",      # Lightest, can often run without dedicated GPU
    "cellpose-sam": "4GB",  # SAM model needs more memory
    "lacss": "2GB",        # JAX-based, moderate
    "samcell": "4GB",      # SAM-based
    "ucell": "2GB",        # FRM-based
    "unifmr": "2GB",       # SwinIR restoration heads
}


def wait_for_service(addr: str, timeout: int = 30) -> bool:
    """Wait for service to become healthy.

    Args:
        addr: Server address (e.g., "127.0.0.1:50051")
        timeout: Maximum wait time in seconds

    Returns:
        True if service became healthy, False if timeout
    """
    for _ in range(timeout):
        try:
            channel = grpc.insecure_channel(addr)
            stub = health_pb2_grpc.HealthStub(channel)
            request = health_pb2.HealthCheckRequest()
            response = stub.Check(request, timeout=2)
            if response.status == health_pb2.HealthCheckResponse.SERVING:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


class DockerService:
    """Handle for a Docker service container."""

    def __init__(
        self,
        service_name: str,
        port: int = 50051,
        build_dir: Optional[Path] = None,
        extra_args: Optional[list] = None,
    ):
        self.service_name = service_name
        self.port = port
        self.build_dir = build_dir or Path(service_name)
        self.extra_args = extra_args or ["--no-token", "--debug"]
        self.container_name = f"biopb-test-{service_name}"
        self._proc: Optional[subprocess.Popen] = None
        self._channel: Optional[grpc.Channel] = None

    def image_exists(self) -> bool:
        """Check if Docker image exists."""
        image_tag = f"{self.service_name}:test"
        result = subprocess.run(
            ["docker", "image", "inspect", image_tag],
            capture_output=True,
        )
        return result.returncode == 0

    def start(self) -> bool:
        """Start the Docker container if image exists.

        If a healthy service is already running on the port, reuse it.
        """
        image_tag = f"{self.service_name}:test"

        # Check if service is already running and healthy
        if wait_for_service(f"127.0.0.1:{self.port}", timeout=2):
            # Service already running, reuse it
            return True

        if not self.image_exists():
            return False

        self._proc = subprocess.Popen(
            [
                "docker", "run", "--rm", "--gpus=all",
                "--name", self.container_name,
                "-p", f"{self.port}:{self.port}",
                image_tag,
                *self.extra_args,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return wait_for_service(f"127.0.0.1:{self.port}")

    def stop(self) -> None:
        """Stop the Docker container (only if we started it)."""
        if self._proc:
            subprocess.run(["docker", "stop", self.container_name], check=False)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def channel(self) -> grpc.Channel:
        """Get gRPC channel to the service."""
        if self._channel is None:
            self._channel = grpc.insecure_channel(
                f"127.0.0.1:{self.port}",
                options=_GRPC_OPTIONS,
            )
        return self._channel

    def detection_stub(self) -> proto.ObjectDetectionStub:
        """Get ObjectDetection stub."""
        return proto.ObjectDetectionStub(self.channel())

    def process_stub(self) -> proto.ProcessImageStub:
        """Get ProcessImage stub."""
        return proto.ProcessImageStub(self.channel())


@pytest.fixture(scope="session")
def test_image():
    """Load standard test image."""
    from tests.utils.image_utils import load_test_image
    return load_test_image()


@pytest.fixture
def test_image_2d():
    """Load a 2D test image with cells."""
    from tests.utils.image_utils import load_test_image
    return load_test_image()


@pytest.fixture
def test_image_multichannel():
    """Generate a multi-channel test image."""
    from tests.utils.image_utils import generate_multichannel_image
    return generate_multichannel_image(512, 512, 3)


@pytest.fixture
def test_image_3d():
    """Generate a 3D test image stack."""
    from tests.utils.image_utils import generate_3d_stack
    return generate_3d_stack(10, 256, 256, 1)


@pytest.fixture
def detection_request_factory():
    """Factory to create DetectionRequest from numpy array."""
    def _create(image: np.ndarray, **settings):
        image_data = serialize_from_numpy_to_image_data(image)
        return proto.DetectionRequest(
            image_data=image_data,
            detection_settings=proto.DetectionSettings(**settings),
        )
    return _create


@pytest.fixture
def process_request_factory():
    """Factory to create ProcessRequest from numpy array."""
    def _create(image: np.ndarray, op_name: str = "", **kwargs):
        image_data = serialize_from_numpy_to_image_data(image)
        return proto.ProcessRequest(
            image_data=image_data,
            op_name=op_name,
        )
    return _create


# Service fixtures - these launch Docker containers
# Use @pytest.mark.service("cellpose") to select which service to test

@pytest.fixture(scope="session")
def cellpose_service():
    """Launch cellpose service for testing.

    Cellpose is the lightest service and can often run on machines
    with limited GPU memory or even CPU-only (slow but functional).

    Requires pre-built image: cellpose:test
    """
    service = DockerService("cellpose")
    if not service.image_exists():
        pytest.skip("Image cellpose:test not found - build it first with: docker build -t cellpose:test cellpose/")
    if not service.start():
        pytest.skip("Failed to start cellpose service")
    yield service
    service.stop()


@pytest.fixture(scope="session")
def cellpose_sam_service():
    """Launch cellpose-sam service for testing.

    Requires ~4GB GPU memory.
    Requires pre-built image: cellpose-sam:test
    """
    service = DockerService("cellpose-sam")
    if not service.image_exists():
        pytest.skip("Image cellpose-sam:test not found - build it first")
    if not service.start():
        pytest.skip("Failed to start cellpose-sam service")
    yield service
    service.stop()


@pytest.fixture(scope="session")
def lacss_service():
    """Launch lacss service for testing.

    Requires ~2GB GPU memory.
    Requires pre-built image: lacss:test
    """
    service = DockerService("lacss")
    if not service.image_exists():
        pytest.skip("Image lacss:test not found - build it first")
    if not service.start():
        pytest.skip("Failed to start lacss service")
    yield service
    service.stop()


@pytest.fixture(scope="session")
def samcell_service():
    """Launch samcell service for testing.

    Requires ~4GB GPU memory.
    Requires pre-built image: samcell:test
    """
    service = DockerService("samcell")
    if not service.image_exists():
        pytest.skip("Image samcell:test not found - build it first")
    if not service.start():
        pytest.skip("Failed to start samcell service")
    yield service
    service.stop()


@pytest.fixture(scope="session")
def ucell_service():
    """Launch ucell service for testing.

    Requires ~2GB GPU memory.
    Requires pre-built image: ucell:test
    """
    service = DockerService("ucell")
    if not service.image_exists():
        pytest.skip("Image ucell:test not found - build it first")
    if not service.start():
        pytest.skip("Failed to start ucell service")
    yield service
    service.stop()


@pytest.fixture(scope="session")
def unifmr_service():
    """Launch unifmr (UNiFMIR restoration) service for testing.

    Requires ~2GB GPU memory.
    Requires pre-built image: unifmr:test
    """
    service = DockerService("unifmr")
    if not service.image_exists():
        pytest.skip("Image unifmr:test not found - build it first")
    if not service.start():
        pytest.skip("Failed to start unifmr service")
    yield service
    service.stop()


# Convenience fixtures for direct channel/stub access

@pytest.fixture
def cellpose_channel(cellpose_service):
    """Get gRPC channel to cellpose service."""
    return cellpose_service.channel()


@pytest.fixture
def cellpose_detection_stub(cellpose_service):
    """Get ObjectDetection stub for cellpose."""
    return cellpose_service.detection_stub()


@pytest.fixture
def cellpose_process_stub(cellpose_service):
    """Get ProcessImage stub for cellpose."""
    return cellpose_service.process_stub()