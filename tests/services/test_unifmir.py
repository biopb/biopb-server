"""Tests for the unifmir (UNiFMIR restoration) service.

UNiFMIR is a restoration model (image -> image), so the suite is built around
``ProcessImage.Run`` rather than the detection-centric ``ServiceTestBase``.
``RunDetection`` is expected to be unimplemented.
"""

import os
import sys

import numpy as np
import pytest

import grpc
import biopb.image as proto
from biopb.image.utils import serialize_from_numpy_to_image_data, deserialize_image_data
from grpc_health.v1 import health_pb2, health_pb2_grpc

_UNIFMIR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "unifmir")


# --------------------------------------------------------------------------- #
# Unit test: vendored model package runs without a checkpoint or Docker.
# --------------------------------------------------------------------------- #
@pytest.mark.smoke
def test_vendored_swinir_forward():
    """The vendored SwinIR backbone instantiates and 2x-upscales a 2D tensor."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    pytest.importorskip("einops")

    sys.path.insert(0, os.path.abspath(_UNIFMIR_DIR))
    try:
        from model.swinir import swinir as SwinIR
    finally:
        sys.path.pop(0)

    model = SwinIR(upscale=2, in_chans=1).eval()
    x = torch.zeros(1, 1, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (1, 1, 128, 128)


# --------------------------------------------------------------------------- #
# Service tests (require a pre-built unifmir:test image).
# --------------------------------------------------------------------------- #
class TestUnifmirService:
    @pytest.mark.smoke
    def test_health_check(self, unifmir_service):
        stub = health_pb2_grpc.HealthStub(unifmir_service.channel())
        response = stub.Check(health_pb2.HealthCheckRequest(), timeout=5)
        assert response.status == health_pb2.HealthCheckResponse.SERVING

    @pytest.mark.smoke
    def test_get_op_names(self, unifmir_service):
        from google.protobuf.empty_pb2 import Empty

        stub = unifmir_service.process_stub()
        response = stub.GetOpNames(Empty(), timeout=10)
        assert "sr_factin" in response.names
        assert "denoise_planaria" in response.names
        assert "isotropic_liver" in response.names

    @pytest.mark.integration
    def test_sr_process_roundtrip(self, unifmir_service):
        """SR (sr_factin) returns a finite 2x-upscaled 2D image."""
        stub = unifmir_service.process_stub()
        image = (np.random.default_rng(0).random((64, 64)) * 255).astype(np.float32)

        request_msg = proto.ProcessRequest(
            image_data=serialize_from_numpy_to_image_data(image),
            op_name="sr_factin",
        )
        response = stub.Run(request_msg, timeout=120)

        result = deserialize_image_data(response.image_data)
        assert result.shape == (128, 128)
        assert np.isfinite(np.asarray(result)).all()

    @pytest.mark.integration
    def test_unknown_op_rejected(self, unifmir_service):
        """An unknown op_name yields INVALID_ARGUMENT."""
        stub = unifmir_service.process_stub()
        image = np.zeros((32, 32), dtype=np.float32)

        request_msg = proto.ProcessRequest(
            image_data=serialize_from_numpy_to_image_data(image),
            op_name="does_not_exist",
        )
        with pytest.raises(grpc.RpcError) as exc:
            stub.Run(request_msg, timeout=30)
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    @pytest.mark.integration
    def test_run_detection_unimplemented(self, unifmir_service):
        """RunDetection is not supported for a restoration model."""
        stub = unifmir_service.detection_stub()
        image = np.zeros((32, 32), dtype=np.float32)

        request_msg = proto.DetectionRequest(
            image_data=serialize_from_numpy_to_image_data(image),
        )
        with pytest.raises(grpc.RpcError) as exc:
            stub.RunDetection(request_msg, timeout=30)
        assert exc.value.code() == grpc.StatusCode.UNIMPLEMENTED


# --------------------------------------------------------------------------- #
# Lazy (chunked) path. Needs a cache-enabled container with the tensor (Flight)
# port exposed; the client sends an inline dask array (debug_pickled_array) and
# reads the assembled result back over Flight. Set UNIFMIR_TEST_CPU=1 to run the
# container on CPU (e.g. on GPUs too old for the prebuilt torch wheels).
# --------------------------------------------------------------------------- #
_LAZY_PORT = 50071
_LAZY_TPORT = 8837


@pytest.fixture(scope="module")
def unifmir_cache_service():
    import subprocess
    import time

    if subprocess.run(["docker", "image", "inspect", "unifmir:test"],
                      capture_output=True).returncode != 0:
        pytest.skip("Image unifmir:test not found - build it first")

    name = "biopb-test-unifmir-cache"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    cpu = os.environ.get("UNIFMIR_TEST_CPU")
    gpu_args = [] if cpu else ["--gpus=all"]
    server_args = [
        "--no-token", "--debug",
        "--cache-dir", "/tmp/tcache",
        "--tensor-external-location", f"grpc://127.0.0.1:{_LAZY_TPORT}",
        "--tile-size", "32",
    ] + (["--no-gpu"] if cpu else [])
    proc = subprocess.Popen(
        ["docker", "run", "--rm", "--name", name, *gpu_args,
         "-p", f"{_LAZY_PORT}:50051", "-p", f"{_LAZY_TPORT}:8817",
         "unifmir:test", *server_args],
        # Discard container logs: we never read these pipes (health is checked
        # via `docker exec grpc_health_probe`), and a full pipe buffer would
        # block the chatty --debug container.
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 60
        ok = False
        while time.time() < deadline:
            if subprocess.run(["docker", "exec", name, "grpc_health_probe",
                               "-addr=localhost:50051"], capture_output=True).returncode == 0:
                ok = True
                break
            time.sleep(2)
        if not ok:
            pytest.skip("unifmir cache service failed to become healthy")
        yield {"port": _LAZY_PORT, "tensor_location": f"grpc://127.0.0.1:{_LAZY_TPORT}"}
    finally:
        subprocess.run(["docker", "stop", name], capture_output=True)
        proc.terminate()


def _lazy_run(svc, op, img, chunks, async_result=False):
    """Send a dask array as lazy input; return the assembled numpy result."""
    os.environ.setdefault("BIOPB_SHM_TRANSFER_DISABLED", "1")
    import dask.array as da
    from biopb.tensor.client import make_debug_serialized_tensor
    from biopb.tensor import TensorFlightClient
    from google.protobuf.struct_pb2 import Struct

    stub = proto.ProcessImageStub(grpc.insecure_channel(f"127.0.0.1:{svc['port']}"))
    st = make_debug_serialized_tensor(da.from_array(img, chunks=chunks), array_id="in")
    kwargs = None
    if async_result:
        kwargs = Struct()
        kwargs["async_result"] = True
    resp = stub.Run(
        proto.ProcessRequest(image_data=proto.ImageData(lazy_data=st), op_name=op, kwargs=kwargs),
        timeout=360,
    )
    assert resp.image_data.HasField("lazy_data")
    if async_result:
        TensorFlightClient(svc["tensor_location"]).wait_for_upload_ready_pb(
            resp.image_data.lazy_data, timeout_seconds=180)
    return np.asarray(deserialize_image_data(resp.image_data))


class TestUnifmirLazy:
    @pytest.mark.integration
    def test_lazy_sr_sync(self, unifmir_cache_service):
        """SR over a multi-chunk lazy input assembles to a 2x output."""
        img = (np.random.default_rng(3).random((96, 96)) * 255).astype(np.float32)
        out = _lazy_run(unifmir_cache_service, "sr_factin", img, chunks=(48, 48))
        assert out.shape == (192, 192)
        assert np.isfinite(out).all()

    @pytest.mark.integration
    def test_lazy_stack(self, unifmir_cache_service):
        """Denoising a lazy Z-stack tiles the Y/X plane and keeps shape."""
        img = (np.random.default_rng(4).random((4, 96, 96)) * 1000).astype(np.float32)
        out = _lazy_run(unifmir_cache_service, "denoise_planaria", img, chunks=(4, 48, 48))
        assert out.shape == (4, 96, 96)
        assert np.isfinite(out).all()

    @pytest.mark.integration
    def test_lazy_async(self, unifmir_cache_service):
        """async_result returns a handle; result is readable once READY."""
        img = (np.random.default_rng(5).random((64, 64)) * 255).astype(np.float32)
        out = _lazy_run(unifmir_cache_service, "sr_factin", img, chunks=(32, 32), async_result=True)
        assert out.shape == (128, 128)
        assert np.isfinite(out).all()
