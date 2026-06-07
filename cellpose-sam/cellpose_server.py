import logging
import threading

import biopb.image as proto
import dask.array as da
import numpy as np
import typer
from google.protobuf.struct_pb2 import Struct

from cellpose import models, io
from biopb_image_base import (
    decode_image_data,
    encode_image,
    BiopbServicerBase,
    run_server,
    ensure_eager,
    parse_kwargs,
    validate_kwargs,
)
# Shared border-region reconciliation for lazy/tiled input (issue #1): single
# source of truth in biopb_image_base, formerly duplicated as local modules.
from biopb_image_base import stitch, dynamics_local

app = typer.Typer(pretty_exceptions_enable=False)

logger = logging.getLogger(__name__)

_TARGET_CELL_SIZE = 30
_NITER = 200  # cellpose's default Euler-integration steps (eval niter=None ~= 200)

# Default kwargs for the cellpose(-SAM) model.
_DEFAULT_KWARGS = {
    "diameter": 0.0,            # 0 => auto-estimate (cellpose eval diameter=None)
    "cellprob_threshold": 0.0,  # matches cellpose eval default
    "min_area": 0,              # post-stitch small-instance removal (lazy path)
    "async_result": False,      # lazy input only; control flag, popped before model call
}

# Validation schema for cellpose kwargs.
_KWARGS_SCHEMA = {
    "diameter": {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1000.0,
        "description": "Cell diameter in pixels (0 = auto-estimate).",
    },
    "cellprob_threshold": {
        "type": "number",
        "minimum": -6.0,
        "maximum": 6.0,
        "description": "Cell probability logit threshold.",
    },
    "min_area": {
        "type": "int",
        "minimum": 0,
        "description": "Minimum cell area in pixels (lazy path).",
    },
    "async_result": {
        "type": "bool",
        "description": "Lazy input only: return the result tensor handle "
                       "immediately and compute in the background. The client "
                       "must poll upload status (wait_for_upload_ready_pb) "
                       "before reading.",
    },
}


def _to_eval_kwargs(kwargs: dict) -> dict:
    """Translate the validated kwargs dict into cellpose ``model.eval`` kwargs.

    Drops control keys (``min_area``, ``async_result``) and maps the ``0.0``
    diameter sentinel to "auto" (omitted, so cellpose estimates it).
    """
    eval_kwargs = {"cellprob_threshold": kwargs["cellprob_threshold"]}
    if kwargs.get("diameter", 0.0):  # 0 => auto-estimate
        eval_kwargs["diameter"] = kwargs["diameter"]
    return eval_kwargs


def process_input(request: proto.DetectionRequest | proto.ProcessRequest):
    """Decode an eager request and return ``(image, kwargs)``.

    Starts from the validated kwargs pipeline (defaults + ``request.kwargs``)
    and lets the legacy ``DetectionSettings`` hints fill in ``diameter`` /
    ``cellprob_threshold`` only when the caller did not pass them explicitly.
    """
    logger.debug(f"Received message of size {request.ByteSize()}")

    image = decode_image_data(request.image_data)
    image = ensure_eager(image)

    kwargs = parse_kwargs(request, _DEFAULT_KWARGS)

    def _passed(key):
        return request.HasField("kwargs") and key in request.kwargs.fields

    if isinstance(request, proto.DetectionRequest):
        settings = request.detection_settings
        pixels = request.image_data.pixels
        physical_size = pixels.physical_size_x or 1

        if not _passed("diameter"):
            if settings.HasField("cell_diameter_hint"):
                kwargs["diameter"] = settings.cell_diameter_hint / physical_size
            elif settings.scaling_hint:
                kwargs["diameter"] = _TARGET_CELL_SIZE / settings.scaling_hint

        if settings.HasField("min_score") and not _passed("cellprob_threshold"):
            kwargs["cellprob_threshold"] = settings.min_score

    errors = validate_kwargs(kwargs, _KWARGS_SCHEMA)
    if errors:
        raise ValueError("Invalid kwargs: " + "; ".join(errors))

    if image.ndim == 2:  # 2D grayscale, add channel dimension
        image = image[..., np.newaxis]

    if image.shape[0] == 1:  # leading singleton dimension
        image = image.squeeze(0)

    logger.debug(f"decoded image {image.shape}")
    logger.debug(f"kwargs: {kwargs}")

    return image, kwargs


def process_result(masks, image):
    import cv2
    from skimage.measure import regionprops

    response = proto.DetectionResponse()

    for rp in regionprops(masks):
        mask = rp.image.astype("uint8")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                polygon=proto.Polygon(points=[proto.Point(x=p[0], y=p[1]) for p in points]),
            ),
        )

        response.detections.append(scored_roi)

    logger.debug(f"Found {len(response.detections)} detections")

    logger.debug(f"Reply with message of size {response.ByteSize()}")

    return response


class CellposeServicer(BiopbServicerBase):

    def __init__(self, model, tile_size: int = 1024, overlap_margin: int = 64):
        # Serialize GPU inference on the explicit _model_lock (a background async
        # lazy job runs after Run returns, outside the base RLock's scope).
        super().__init__(use_lock=False)
        self.model = model
        # Lazy-path chunking. overlap_margin must exceed the largest expected
        # cell diameter for border IDs to stay consistent (see stitch.py).
        self.tile_size = tile_size
        self.overlap_margin = overlap_margin
        self._model_lock = threading.Lock()

    def GetOpNames(self, request, context):
        """Return the available operations and their parameter schemas."""
        with self._server_context(context):
            default_kwargs = Struct()
            default_kwargs.update(_DEFAULT_KWARGS)
            return proto.OpNames(
                names=["cellpose"],
                op_schemas={
                    "cellpose": proto.OpSchema(
                        description="Cellpose-SAM cell segmentation model",
                        default_kwargs=default_kwargs,
                    ),
                }
            )

    def RunDetection(self, request, context):
        with self._server_context(context):
            image, kwargs = process_input(request)

            if image.ndim == 4:
                raise ValueError("3D input not supported. Use the 'ProcessImage' service instead.")

            logger.info(f"call model with image {image.shape}")

            with self._model_lock:
                masks = self.model.eval(image, **_to_eval_kwargs(kwargs))[0]

            logger.info(f"received masks {masks.shape} with {masks.max()} labels.")

            response = process_result(masks, image)

            return response

    def Run(self, request, context):
        with self._server_context(context):
            logger.info(f"Received message of size {request.ByteSize()}")

            image = decode_image_data(request.image_data)

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

            if image.ndim == 2:  # 2D grayscale, add channel dimension
                image = image[..., np.newaxis]
            if image.shape[0] == 1:  # leading singleton dimension
                image = image.squeeze(0)

            eval_kwargs = _to_eval_kwargs(kwargs)
            with self._model_lock:
                if image.ndim == 3:  # 2D (Y, X, C)
                    mask = self.model.eval(image, **eval_kwargs)[0]
                else:  # 3D
                    mask = self.model.eval(
                        image,
                        channel_axis=-1,
                        z_axis=0,
                        do_3D=True,
                        flow3D_smooth=1,
                        **eval_kwargs,
                    )[0]

            logger.info(f"Detected {mask.max()} cells")

            response = proto.ProcessResponse(
                image_data=encode_image(mask),
            )

            logger.debug(f"Reply with message of size {response.ByteSize()}")

            return response

    def _run_lazy(self, image, kwargs, async_result=False) -> proto.ProcessResponse:
        """Tile-wise ProcessImage for a lazy (dask) input, larger than memory.

        Allocates the label-mask output at the co-located tensor cache, then
        runs :func:`stitch.stitch_lazy_segmentation`: each overlapping tile is
        fetched, run through cellpose to get per-pixel flow + cell probability,
        integrated to per-pixel destinations, clustered with cross-tile ID
        inheritance, cropped to its core, and uploaded -- so neither the input
        nor the output is ever fully in memory.

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

        eval_kwargs = _to_eval_kwargs(kwargs)

        def compute_chunk(tile_start, tile_stop):
            ys, xs = tile_start
            ye, xe = tile_stop
            if channel_axis is None:
                tile = image[ys:ye, xs:xe].compute()
            else:
                tile = image[ys:ye, xs:xe, :].compute()
            if tile.ndim == 2:
                tile = tile[..., np.newaxis]
            # Lock guards the shared GPU model (a background async job may run
            # concurrently with another request's inference).
            with self._model_lock:
                _, flows, _ = self.model.eval(tile, **eval_kwargs)  # mask discarded
            dP = flows[1]        # (2, Ly, Lx) flow field, cellpose native scale
            cell_prob = flows[2]  # (Ly, Lx)
            # SCALE = 1.0: cellpose flows[1] already matches the /5.0 convention
            # that dynamics_local.compute_destinations applies internally (ucell
            # uses *4.0 only for its FRM model). Verified by test_lazy_matches_eager.
            inds, p = dynamics_local.compute_destinations(
                dP,
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


@app.command()
def main(
    modeltype: str = "",  # ignored, kept for compatibility
    port: int = 50051,
    workers: int = 10,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: bool | None = None,
    debug: bool = False,
    compression: bool = True,
    gpu: bool = True,
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
    model = models.CellposeModel(gpu=gpu)
    io.logger_setup()

    run_server(
        CellposeServicer(model, tile_size=tile_size, overlap_margin=overlap_margin),
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
