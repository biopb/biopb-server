"""Tests for Cellpose-SAM service."""

import os

import pytest
import numpy as np

import grpc
import biopb.image as proto
from biopb.image.utils import serialize_from_numpy_to_image_data, deserialize_image_data
from google.protobuf.struct_pb2 import Struct
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

        result = deserialize_image_data(response.image_data)
        assert result.shape[:2] == test_image_2d.shape[:2]

    @pytest.mark.integration
    def test_get_op_names(self, request):
        """GetOpNames should return available operations and default kwargs."""
        service = request.getfixturevalue("cellpose_sam_service")
        stub = service.process_stub()

        from google.protobuf.empty_pb2 import Empty
        response = stub.GetOpNames(Empty(), timeout=10)
        assert "cellpose" in response.names
        # The op schema now advertises the validated default kwargs.
        schema = response.op_schemas["cellpose"]
        assert "cellprob_threshold" in schema.default_kwargs.fields

    @pytest.mark.integration
    def test_diameter_kwarg(self, request, test_image_2d):
        """An explicit diameter kwarg is accepted via the kwargs Struct."""
        service = request.getfixturevalue("cellpose_sam_service")
        stub = service.detection_stub()

        kwargs = Struct()
        kwargs.update({"diameter": 30.0})
        request_msg = proto.DetectionRequest(
            image_data=serialize_from_numpy_to_image_data(test_image_2d),
            kwargs=kwargs,
        )
        response = stub.RunDetection(request_msg, timeout=60)
        assert len(response.detections) > 0

    @pytest.mark.integration
    def test_cell_diameter_hint(self, request, test_image_2d):
        """The legacy DetectionSettings.cell_diameter_hint path still works."""
        service = request.getfixturevalue("cellpose_sam_service")
        stub = service.detection_stub()

        request_msg = proto.DetectionRequest(
            image_data=serialize_from_numpy_to_image_data(test_image_2d),
            detection_settings=proto.DetectionSettings(cell_diameter_hint=30.0),
        )
        response = stub.RunDetection(request_msg, timeout=60)
        assert len(response.detections) > 0

    @pytest.mark.contract
    def test_invalid_kwarg_rejected(self, request, test_image_2d):
        """An out-of-range kwarg yields INVALID_ARGUMENT."""
        service = request.getfixturevalue("cellpose_sam_service")
        stub = service.process_stub()

        kwargs = Struct()
        kwargs.update({"cellprob_threshold": 99.0})  # schema max is 6.0
        request_msg = proto.ProcessRequest(
            image_data=serialize_from_numpy_to_image_data(test_image_2d),
            kwargs=kwargs,
        )
        with pytest.raises(grpc.RpcError) as exc:
            stub.Run(request_msg, timeout=30)
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


# --------------------------------------------------------------------------- #
# Lazy (chunked) path. Needs a cache-enabled container with the tensor (Flight)
# port exposed; the client sends an inline dask array (debug_pickled_array) and
# reads the assembled label mask back over Flight. Set CELLPOSE_SAM_TEST_CPU=1
# to run the container on CPU.
# --------------------------------------------------------------------------- #
_LAZY_PORT = 50061
_LAZY_TPORT = 8827


@pytest.fixture(scope="module")
def cellpose_sam_cache_service():
    import subprocess
    import time

    if subprocess.run(["docker", "image", "inspect", "cellpose-sam:test"],
                      capture_output=True).returncode != 0:
        pytest.skip("Image cellpose-sam:test not found - build it first")

    name = "biopb-test-cellpose-sam-cache"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    cpu = os.environ.get("CELLPOSE_SAM_TEST_CPU")
    gpu_args = [] if cpu else ["--gpus=all"]
    server_args = [
        "--no-token", "--debug",
        "--cache-dir", "/tmp/tcache",
        "--tensor-external-location", f"grpc://127.0.0.1:{_LAZY_TPORT}",
        "--tile-size", "256",
    ] + (["--no-gpu"] if cpu else [])
    proc = subprocess.Popen(
        ["docker", "run", "--rm", "--name", name, *gpu_args,
         "-p", f"{_LAZY_PORT}:50051", "-p", f"{_LAZY_TPORT}:8817",
         "cellpose-sam:test", *server_args],
        # Discard container logs: we never read these pipes (health is checked
        # via `docker exec grpc_health_probe`), and a full pipe buffer would
        # block the chatty --debug container.
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 120
        ok = False
        while time.time() < deadline:
            if subprocess.run(["docker", "exec", name, "grpc_health_probe",
                               "-addr=localhost:50051"], capture_output=True).returncode == 0:
                ok = True
                break
            time.sleep(2)
        if not ok:
            pytest.skip("cellpose-sam cache service failed to become healthy")
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


def _eager_run(svc, img):
    """Send the same image inline (eager) and return the label mask."""
    stub = proto.ProcessImageStub(grpc.insecure_channel(f"127.0.0.1:{svc['port']}"))
    resp = stub.Run(
        proto.ProcessRequest(image_data=serialize_from_numpy_to_image_data(img)),
        timeout=360,
    )
    return np.asarray(deserialize_image_data(resp.image_data))


def _tiled_test_image(test_image_2d):
    """A multi-core (>256 per axis) 2D image built from the real test image."""
    img = np.asarray(test_image_2d)
    reps = tuple(max(1, -(-512 // s)) for s in img.shape[:2])  # ceil-div to >=512
    return np.tile(img, reps)


class TestCellposeSamLazy:
    @pytest.mark.integration
    def test_lazy_process_2d(self, cellpose_sam_cache_service, test_image_2d):
        """A multi-core lazy input assembles to a same-shape label mask."""
        img = _tiled_test_image(test_image_2d)
        out = _lazy_run(cellpose_sam_cache_service, "cellpose", img, chunks=(256, 256))
        assert out.shape == img.shape[:2]
        assert np.issubdtype(out.dtype, np.integer)
        assert out.max() > 0  # at least one instance found

    @pytest.mark.integration
    def test_lazy_async(self, cellpose_sam_cache_service, test_image_2d):
        """async_result returns a handle; result is readable once READY."""
        img = _tiled_test_image(test_image_2d)
        out = _lazy_run(cellpose_sam_cache_service, "cellpose", img,
                        chunks=(256, 256), async_result=True)
        assert out.shape == img.shape[:2]
        assert out.max() > 0

    @pytest.mark.integration
    def test_lazy_matches_eager(self, cellpose_sam_cache_service, test_image_2d):
        """Lazy tiled segmentation finds ~the same number of cells as eager.

        This is the empirical check on the flow scale passed to
        ``dynamics_local.compute_destinations`` (see _run_lazy). A grossly wrong
        scale fragments or merges cells and blows up / collapses the count.
        """
        img = _tiled_test_image(test_image_2d)
        n_eager = int(_eager_run(cellpose_sam_cache_service, img).max())
        n_lazy = int(_lazy_run(cellpose_sam_cache_service, "cellpose", img,
                               chunks=(256, 256)).max())
        assert n_eager > 0 and n_lazy > 0
        # Tiling/border reconciliation differs from whole-image dynamics, so
        # allow a moderate tolerance; a wrong flow scale fails this by a lot.
        assert abs(n_lazy - n_eager) <= max(5, 0.25 * n_eager)
