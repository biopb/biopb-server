"""UCell gRPC server using biopb_image_base utilities."""

import logging
import threading
from pathlib import Path

import biopb.image as proto
import dask.array as da
import numpy as np
import torch
import typer
from google.protobuf.struct_pb2 import Struct
from ml_collections import ConfigDict
from biopb.image.utils import get_image_data_dim_labels, normalize_array_dims, deserialize_image_data

from ucell.dynamics import compute_masks
from ucell.frm import FRMWrapper
from ucell.utils import pad_channel, patcherize

# Sibling, import-light service modules (top-level: they sit next to this file
# at $HOME in the container; tests add the dir to sys.path).
import stitch
import dynamics_local

from biopb_image_base import (
    BiopbServicerBase,
    encode_image,
    run_server,
    parse_kwargs,
    validate_kwargs,
    ensure_eager,
)

app = typer.Typer(pretty_exceptions_enable=False)

logger = logging.getLogger(__name__)

# Default kwargs for ucell model
_DEFAULT_KWARGS = {
    "task_id": 0,
    "cellprob_threshold": -0.2,
    "min_area": 5,
    "async_result": False,
}

# Validation schema for ucell kwargs
_KWARGS_SCHEMA = {
    "task_id": {
        "type": "int",
        "minimum": 0,
        "description": "Task ID for multi-task models",
    },
    "cellprob_threshold": {
        "type": "number",
        "minimum": -6.0,
        "maximum": 6.0,
        "description": "Cell probability logit threshold",
    },
    "min_area": {
        "type": "int",
        "minimum": 0,
        "description": "Minimum cell area",
    },
    "async_result": {
        "type": "bool",
        "description": "Lazy input only: return the result tensor handle "
                       "immediately and compute in the background. The client "
                       "must poll upload status (wait_for_upload_ready_pb) "
                       "before reading.",
    },
}

_NITER = 500  # Integration steps for mask computation

def get_config():
    """Return default ucell model configuration."""
    config = ConfigDict()
    config.name = "ucell"
    config.seed = 42
    config.image_size = 256
    config.task_id = 0

    config.model = ConfigDict()
    config.model.patch_size = 8
    config.model.forward_dtype = "bfloat16"
    config.model.pos_emb = "rope"
    config.model.hidden_size = 1024
    config.model.num_z_tokens = 64
    config.model.num_task_emb_tokens = 64
    config.model.depth = 2
    config.model.num_heads = config.model.hidden_size // 64
    config.model.num_tasks = 5
    config.model.seq_len = (config.image_size // config.model.patch_size) ** 2
    config.model.H_cycles = 1
    config.model.L_cycles = 21
    config.halt_max_steps = 1

    # LoRA configuration (disabled by default)
    config.lora = ConfigDict()
    config.lora.rank = 0
    config.lora.alpha = 1.0
    config.lora.dropout = 0.0
    config.lora.target_modules = ["SwiGLU"]

    return config


def format_image(img: np.ndarray) -> np.ndarray:
    """Normalize and pad image channels for ucell model."""
    img = img / (img.max() + 1e-5)
    img = pad_channel(img)
    return img

def get_image_data(image_data: proto.ImageData):
    """Decode and normalize image data from the request."""
    image = deserialize_image_data(image_data)
    dim_labels = get_image_data_dim_labels(image_data)
    if dim_labels is not None:
        image = normalize_array_dims(
            image, dim_labels, ["Z", "Y", "X", "C"],
        )
    elif image.ndim in (2, 3, 4):
        logger.warning("Input image is missing dim_labels; assuming (Z)YX(C) with optional leading singleton Z and trailing C.")
        if image.ndim == 2:
            image = image[None, :, :, None]  # Add Z, C dim
        elif image.ndim == 3:
            if image.shape[-1] > 3:  # Heuristic: likely ZYX
                image = image[:, :, :, None]  # Add C dim
            else:
                image = image[None, :, :, :]  # Add Z dim
    else:
        raise ValueError(f"Input image has {image.ndim} dims but no dim_labels; expected 2D, 3D, or 4D with (Z)YX(C) format.")
    return image

def process_input(request: proto.DetectionRequest):
    """Process input request and return image and kwargs."""
    image = get_image_data(request.image_data)
    image = ensure_eager(image)

    # Handle 2D images
    if image.shape[0] == 1:
        image = image.squeeze(0)

    # Normalize and pad channels
    image = format_image(image)

    # Parse kwargs
    kwargs = parse_kwargs(request, _DEFAULT_KWARGS)

    # Validate kwargs
    errors = validate_kwargs(kwargs, _KWARGS_SCHEMA)
    if errors:
        raise ValueError("Invalid kwargs: " + "; ".join(errors))

    return image, kwargs


def process_result(mask: np.ndarray, image: np.ndarray) -> proto.DetectionResponse:
    """Convert mask to DetectionResponse with polygon ROIs."""
    import cv2
    from skimage.measure import regionprops

    response = proto.DetectionResponse()

    for rp in regionprops(mask):
        mask_region = rp.image.astype("uint8")
        contours, _ = cv2.findContours(mask_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        # Use the largest contour
        contour = max(contours, key=len)

        # Need at least 3 points for a valid polygon
        if len(contour) < 3:
            continue

        # Extract points and shift to global coordinates
        points = contour.squeeze(1)  # Shape: (N, 2)
        points = points + np.array([rp.bbox[1], rp.bbox[0]])
        points = points - 0.5

        scored_roi = proto.ScoredROI(
            score=1.0,
            roi=proto.ROI(
                polygon=proto.Polygon(
                    points=[proto.Point(x=p[0], y=p[1]) for p in points]
                ),
            ),
        )

        response.detections.append(scored_roi)

    logger.debug(f"Found {len(response.detections)} detections")

    return response


def compute_instance_masks(flow: np.ndarray, cell_prob: np.ndarray, kwargs: dict, device: torch.device) -> np.ndarray:
    """Compute instance masks from flow and cell probability."""
    mask = compute_masks(
        flow * 4.0,
        cell_prob,
        niter=_NITER,
        cellprob_threshold=kwargs["cellprob_threshold"],
        flow_threshold=0,
        min_size=kwargs["min_area"],
        max_size_fraction=0.4,
        device=device,
    )

    return mask


class UCellServicer(BiopbServicerBase):
    """UCell servicer implementing ObjectDetection and ProcessImage services."""

    def __init__(self, model, config, device: torch.device,
                 tile_size: int = 1024, overlap_margin: int = 64):
        super().__init__(use_lock=False)
        self.model = model
        self.config = config
        self.device = device
        # Lazy-path chunking. overlap_margin must exceed the largest expected
        # cell diameter for border IDs to stay consistent (see stitch.py).
        self.tile_size = tile_size
        self.overlap_margin = overlap_margin
        # Serializes ProcessImage GPU inference. An async lazy job runs in a
        # background thread after Run returns, so this lock keeps it from racing
        # a concurrent inference on the single shared model.
        self._model_lock = threading.Lock()

    def RunDetection(self, request, context):
        with self._server_context(context):
            logger.info(f"Received message of size {request.ByteSize()}")

            image, kwargs = process_input(request)

            logger.info(f"Received image {image.shape}")

            # Run inference with patcherize for large images
            predict_fn = patcherize(self.model.predict, GS=self.config.image_size)
            with torch.device(self.device):
                output = predict_fn(image, kwargs["task_id"])

            # output shape: (H, W, 3) - flow[:2], cell_prob[2]
            flow = np.moveaxis(output[:, :, :2], -1, 0)  # (2, H, W)
            cell_prob = output[:, :, 2]

            mask = compute_instance_masks(flow, cell_prob, kwargs, self.device)

            response = process_result(mask, image)

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response

    def Run(self, request, context):
        with self._server_context(context):
            logger.info(f"Received message of size {request.ByteSize()}")

            image = get_image_data(request.image_data)

            kwargs = parse_kwargs(request, _DEFAULT_KWARGS)
            errors = validate_kwargs(kwargs, _KWARGS_SCHEMA)
            if errors:
                raise ValueError("Invalid kwargs: " + "; ".join(errors))

            # Control parameter, not a model kwarg.
            async_result = bool(kwargs.pop("async_result", False))

            # Lazy (dask) input may be larger than memory: process it tile-by-tile
            # and stream a lazy label-mask result back via the tensor cache.
            if isinstance(image, da.Array):
                return self._run_lazy(image, kwargs, async_result=async_result)

            image = ensure_eager(image)

            if image.shape[0] == 1:  # 2D
                image = image.squeeze(0)

            image = format_image(image)

            logger.info(f"Received image {image.shape}")

            predict_fn = patcherize(self.model.predict, GS=self.config.image_size)
            with self._model_lock, torch.device(self.device):
                output = predict_fn(image, kwargs["task_id"])

            flow = np.moveaxis(output[:, :, :2], -1, 0)
            cell_prob = output[:, :, 2]

            mask = compute_instance_masks(flow, cell_prob, kwargs, self.device)

            logger.info(f"Detected {mask.max()} cells")

            response = proto.ProcessResponse(
                image_data=encode_image(mask.astype(np.uint16)),
            )

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response

    def _run_lazy(self, image, kwargs, async_result=False) -> proto.ProcessResponse:
        """Tile-wise ProcessImage for a lazy (dask) input, larger than memory.

        Allocates the label-mask output at the co-located tensor cache, then
        runs :func:`stitch.stitch_lazy_segmentation`: each overlapping tile is
        fetched, run through the model + flow integration to get per-pixel
        destinations, clustered with cross-tile ID inheritance, cropped to its
        core, and uploaded -- so neither the input nor the output is ever fully
        in memory.

        With ``async_result=False`` (default) the stitch runs inline and the
        returned ``lazy_data`` is a completed result. With ``async_result=True``
        the stitch runs on a background thread and the registration handle is
        returned immediately (avoiding a long-held RPC / client disconnect); the
        client must poll ``wait_for_upload_ready_pb`` before reading.
        """
        if self._tensor_cache is None:
            raise ValueError(
                "Lazy (dask) input requires a tensor cache to write the result. "
                "Start the server with --cache-dir (or set TENSOR_SERVER_URL)."
            )

        # Collapse a leading singleton Z (2D images arrive as (1, Y, X[, C])).
        if image.ndim >= 3 and image.shape[0] == 1:
            image = image[0]

        if image.ndim == 2:
            full_shape, channel_axis = tuple(image.shape), None
        elif image.ndim == 3:
            full_shape, channel_axis = tuple(image.shape[:2]), 2
        else:
            raise ValueError("Lazy processing currently supports 2D images only.")

        margin = self.overlap_margin
        core_shape = tuple(
            stitch.uniform_core(full_shape[i], self.tile_size) for i in range(2)
        )
        logger.info(
            f"Lazy ProcessImage: image {image.shape}, core {core_shape}, margin {margin}"
        )

        from biopb.tensor import ChunkBounds

        template = da.zeros(full_shape, chunks=core_shape, dtype=np.int32)
        registration = self._tensor_cache.create_array(
            source_name=None, dim_labels=["Y", "X"], array_template=template
        )
        source_id = registration.tensor_descriptor.array_id

        predict_fn = patcherize(self.model.predict, GS=self.config.image_size)

        def compute_chunk(tile_start, tile_stop):
            ys, xs = tile_start
            ye, xe = tile_stop
            if channel_axis is None:
                tile = image[ys:ye, xs:xe].compute()
            else:
                tile = image[ys:ye, xs:xe, :].compute()
            tile = format_image(tile)
            # Lock guards the shared GPU model (a background async job may run
            # concurrently with another request's inference).
            with self._model_lock, torch.device(self.device):
                output = predict_fn(tile, kwargs["task_id"])
            flow = np.moveaxis(output[:, :, :2], -1, 0)
            cell_prob = output[:, :, 2]
            # Mirror compute_instance_masks: dP = flow * 4.0, flow_threshold off.
            inds, p = dynamics_local.compute_destinations(
                flow * 4.0,
                cell_prob,
                cellprob_threshold=kwargs["cellprob_threshold"],
                niter=_NITER,
            )
            return inds, p

        def write_core(core_start, core_stop, labels):
            bounds = ChunkBounds(start=list(core_start), stop=list(core_stop))
            self._tensor_cache.upload_array_chunks(
                source_id, bounds, np.ascontiguousarray(labels, dtype=np.int32)
            )

        def run_stitch():
            return stitch.stitch_lazy_segmentation(
                full_shape, core_shape, margin, compute_chunk, write_core,
                min_area=kwargs["min_area"],
            )

        if async_result:
            # Return the registration handle now; compute + upload in the
            # background. The source flips to READY once every core is uploaded.
            def worker():
                try:
                    n_ids = run_stitch()
                    logger.info(f"[async] lazy segmentation produced {n_ids} "
                                f"instances for source {source_id}")
                except Exception:
                    # No FAILED-marking API on the cache; a crashed job leaves
                    # the source UPLOADING, so the client's poll will time out.
                    logger.exception(f"[async] lazy segmentation FAILED for "
                                     f"source {source_id}; client poll will time out")

            threading.Thread(target=worker, name=f"lazy-{source_id}").start()
            logger.info(f"Lazy ProcessImage (async): returning handle for "
                        f"source {source_id}; computing in background")
            return proto.ProcessResponse(
                image_data=proto.ImageData(lazy_data=registration),
            )

        n_ids = run_stitch()
        logger.info(f"Lazy segmentation produced {n_ids} instances")

        serialized = self._tensor_cache.to_serialized_tensor(source_id)
        return proto.ProcessResponse(
            image_data=proto.ImageData(lazy_data=serialized),
        )

    def GetOpNames(self, request, context):
        """Return the available operations and their parameter schemas."""
        with self._server_context(context):
            default_kwargs = Struct()
            default_kwargs.update(_DEFAULT_KWARGS)
            return proto.OpNames(
                names=["ucell"],
                op_schemas={
                    "ucell": proto.OpSchema(
                        description="UCell cell segmentation model (FRM-based)",
                        default_kwargs=default_kwargs,
                    ),
                }
            )


def load_model(modelpath: Path, config, device: torch.device):
    """Load and initialize the UCell model."""
    model = FRMWrapper(config).eval()

    default_lora_scaling = None
    if hasattr(config, "lora") and config.lora.rank > 0:
        default_lora_scaling = config.lora.alpha / config.lora.rank

    model.load_checkpoint(
        str(modelpath),
        default_lora_scaling=default_lora_scaling,
    )

    model = model.inner.to(device)

    if device.type == "cuda":
        model = torch.compile(model)

    logger.info(f"Loaded model from {modelpath}")

    return model


@app.command()
def main(
    modelpath: Path = typer.Option("./model.pt", help="Path to the model checkpoint"),
    port: int = 50051,
    workers: int = 10,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: bool | None = None,
    debug: bool = False,
    compression: bool = True,
    gpu: bool = True,
    small_model: bool = False,
    cache_dir: str | None = typer.Option(
        None, help="Directory for the embedded tensor cache. Required to accept "
                   "lazy (dask) input / return lazy output."),
    cache_size: str = "32GB",
    tensor_port: int = 8817,
    tensor_external_location: str | None = None,
    tile_size: int = typer.Option(
        1024, help="Target core (non-overlap) tile size for lazy processing."),
    overlap_margin: int = typer.Option(
        64, help="Tile overlap; must exceed the largest expected cell diameter."),
):
    config = get_config()
    if small_model:
        config.model.hidden_size = 768

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model = load_model(modelpath, config, device)

    logger.info(f"Default backend: {device}")

    if device.type == "cpu":
        logger.warning("WARNING: No GPU configuration. This might be very slow...")

    run_server(
        UCellServicer(model, config, device,
                      tile_size=tile_size, overlap_margin=overlap_margin),
        port=port,
        workers=workers,
        ip=ip,
        local=local,
        token=token,
        log_level="DEBUG" if debug else "INFO",
        compression=compression,
        cache_dir=cache_dir,
        cache_size=cache_size,
        tensor_port=tensor_port,
        tensor_external_location=tensor_external_location,
    )


if __name__ == "__main__":
    app()
