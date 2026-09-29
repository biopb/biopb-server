"""Shared pytest fixtures for biopb-server tests.

Every service is an Ops server, run in a container that shares the host's network.
Note: Most services require GPU. On machines without sufficient GPU,
tests will be skipped. Cellpose is the most lightweight and can often
run on machines with limited GPU memory.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Optional

import grpc
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc

import biopb.image as proto


# Default gRPC options for large messages
_GRPC_OPTIONS = [
    ("grpc.max_receive_message_length", 256 * 1024 * 1024),  # 256MB
    ("grpc.max_send_message_length", 256 * 1024 * 1024),
]

# The line a server logs once it is bound: "serving cellpose on 127.0.0.1:41235".
_SERVING = re.compile(r"serving .+ on [\d.]+:(\d+)")


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
    """Handle for an Ops service container, run from the image ``<name>:test``.

    The container shares the host's network and binds 127.0.0.1, where an Ops
    server takes no token. It is started on port 0, so the kernel picks a free
    port and no two services (or two test sessions) can collide; the port is
    read back from the line the server logs once it is bound.

    The container gets the GPUs (``--gpus=all``) unless ``BIOPB_TEST_CPU`` is
    set, for a machine without a GPU the image's torch supports. Inference on
    the CPU is slow.
    """

    def __init__(self, service_name: str, extra_args: Optional[list] = None):
        self.service_name = service_name
        self.extra_args = extra_args or []
        self.port: Optional[int] = None
        self.container_name = f"biopb-test-{service_name}-{os.getpid()}"
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

    def _logged_port(self, timeout: int) -> Optional[int]:
        """The port the container reports it is serving on, or None."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                return None  # the container exited before it bound
            logs = subprocess.run(
                ["docker", "logs", self.container_name],
                capture_output=True, text=True,
            )
            match = _SERVING.search(logs.stdout + logs.stderr)
            if match:
                return int(match.group(1))
            time.sleep(1)
        return None

    def start(self, timeout: int = 120) -> bool:
        """Start the Docker container if the image exists."""
        if not self.image_exists():
            return False

        subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True)
        gpu_args = [] if os.environ.get("BIOPB_TEST_CPU") else ["--gpus=all"]
        # Logs are read back with `docker logs`: a pipe nobody drains would
        # block the container once it filled.
        self._proc = subprocess.Popen(
            [
                "docker", "run", "--rm", *gpu_args, "--network", "host",
                "--name", self.container_name,
                f"{self.service_name}:test",
                "--host", "127.0.0.1", "--port", "0",
                *self.extra_args,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.port = self._logged_port(timeout)
        if self.port is None:
            self.stop()
            return False
        return wait_for_service(f"127.0.0.1:{self.port}")

    def stop(self) -> None:
        """Stop the Docker container."""
        if self._proc:
            subprocess.run(["docker", "stop", self.container_name], check=False,
                           capture_output=True)
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


def _service_fixtures(name: str, vram: str, note: str = ""):
    """The fixtures for one service: the container, its channel, its Ops stub.

    They are named after the service (dashes to underscores): ``<name>_service``
    (session-scoped), ``<name>_channel`` and ``<name>_ops_stub``.
    """
    ident = name.replace("-", "_")

    @pytest.fixture(scope="session")
    def service():
        service = DockerService(name)
        if not service.image_exists():
            pytest.skip(f"Image {name}:test not found - build it first with: docker build -t {name}:test {name}/")
        if not service.start():
            pytest.skip(f"Failed to start {name} service")
        yield service
        service.stop()

    @pytest.fixture
    def channel(request):
        return request.getfixturevalue(f"{ident}_service").channel()

    @pytest.fixture
    def ops_stub(request):
        return proto.OpsStub(request.getfixturevalue(f"{ident}_service").channel())

    service.__doc__ = f"Launch the {name} service for testing. Needs ~{vram} GPU memory. {note}".strip()
    return {f"{ident}_service": service, f"{ident}_channel": channel, f"{ident}_ops_stub": ops_stub}


# Approximate VRAM each service needs.
globals().update(
    {
        **_service_fixtures("cellpose", "1GB", "The lightest; often runs on a small GPU."),
        **_service_fixtures("cellpose-sam", "4GB"),
        **_service_fixtures("samcell", "4GB"),
        **_service_fixtures("ucell", "2GB"),
        **_service_fixtures("unifmir", "2GB"),
    }
)
