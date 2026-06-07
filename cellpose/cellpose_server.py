import logging
import threading

import biopb.image as proto
import dask.array as da
import numpy as np
import typer
from google.protobuf.struct_pb2 import Struct

from cellpose import models
from biopb.image.utils import deserialize_image_data, get_image_data_dim_labels, normalize_array_dims
from biopb_image_base import encode_image, BiopbServicerBase, run_server, parse_kwargs, validate_kwargs, ensure_eager
# Shared border-region reconciliation for lazy/tiled input (issue #1): single
# source of truth in biopb_image_base, formerly duplicated as local modules.
from biopb_image_base import stitch, dynamics_local

app = typer.Typer(pretty_exceptions_enable=False)

logger = logging.getLogger(__name__)

_TARGET_CELL_SIZE=30

# Default kwargs for cellpose model
_DEFAULT_KWARGS = {
    "channels": [0, 0],
    "diameter": 30.0,
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "normalize": True,
    "invert": False,
    "min_size": 15,
    "async_result": False,
}

# Validation schema for cellpose kwargs
_CELLPOSE_kwargs_SCHEMA = {
    "channels": {
        "type": "array",
        "item_type": "int",
        "min_length": 2,
        "max_length": 2,
        "description": "Channel specification [cytoplasm, nucleus]",
    },
    "diameter": {
        "type": "number",
        "minimum": 0,
        "description": "Cell diameter in pixels",
    },
    "flow_threshold": {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "description": "Flow error threshold",
    },
    "cellprob_threshold": {
        "type": "number",
        "minimum": -6.0,
        "maximum": 6.0,
        "description": "Cell probability threshold",
    },
    "normalize": {
        "type": "bool",
        "description": "Normalize image intensities",
    },
    "invert": {
        "type": "bool",
        "description": "Invert image pixel intensity",
    },
    "min_size": {
        "type": "int",
        "minimum": 0,
        "description": "Minimum cell size",
    },
    "async_result": {
        "type": "bool",
        "description": "Lazy input only: return the result tensor handle "
                       "immediately and compute in the background. The client "
                       "must poll upload status (wait_for_upload_ready_pb) "
                       "before reading.",
    },
}


def get_image_data(image_data: proto.ImageData):
    """Decode and normalize image data to (Z, Y, X, C).

    Uses the request's dim_labels when present; otherwise falls back to a
    heuristic for (Z)YX(C) with an optional leading singleton Z and trailing C.
    """
    image = deserialize_image_data(image_data)
    dim_labels = get_image_data_dim_labels(image_data)
    if dim_labels is not None:
        image = normalize_array_dims(image, dim_labels, ["Z", "Y", "X", "C"])
    elif image.ndim in (2, 3, 4):
        logger.warning("Input image is missing dim_labels; assuming (Z)YX(C) "
                       "with optional leading singleton Z and trailing C.")
        if image.ndim == 2:
            image = image[None, :, :, None]  # add Z, C
        elif image.ndim == 3:
            if image.shape[-1] > 3:  # heuristic: likely ZYX
                image = image[:, :, :, None]  # add C
            else:
                image = image[None, :, :, :]  # add Z
    else:
        raise ValueError(f"Input image has {image.ndim} dims but no dim_labels; "
                         "expected 2D, 3D, or 4D with (Z)YX(C) format.")
    return image


def process_input(request: proto.DetectionRequest):
    settings = request.detection_settings

    image = get_image_data(request.image_data)
    image = ensure_eager(image)

    pixels = request.image_data.pixels
    physical_size = pixels.physical_size_x or 1

    # Start with default kwargs
    kwargs = _DEFAULT_KWARGS.copy()

    # Override diameter from detection_settings if provided
    if settings.HasField("cell_diameter_hint"):
        kwargs["diameter"] = settings.cell_diameter_hint / physical_size
    elif settings.scaling_hint:
        kwargs["diameter"] = _TARGET_CELL_SIZE / (settings.scaling_hint or 1.0)

    # Auto-detect channels if not specified in kwargs
    if image.shape[-1] > 1:
        kwargs["channels"] = [1, 2]
    else:
        kwargs["channels"] = [0, 0]

    # Merge with kwargs from request (if provided)
    kwargs = parse_kwargs(request, kwargs)

    # Validate kwargs
    errors = validate_kwargs(kwargs, _CELLPOSE_kwargs_SCHEMA)
    if errors:
        raise ValueError("Invalid kwargs: " + "; ".join(errors))

    # async_result is a lazy-output control flag (ProcessImage only); it has no
    # meaning for RunDetection and is not a Cellpose model kwarg -- drop it so it
    # is not forwarded to model.eval().
    kwargs.pop("async_result", None)

    if image.shape[0] == 1: # 2D
        image = image.squeeze(0)

    return image, kwargs


def process_result(preds, image):
    import cv2
    from skimage.measure import regionprops

    response = proto.DetectionResponse()

    try:
        masks, flows, styles, _ = preds
    except:
        masks, flows, styles = preds

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

    return response


class CellposeServicer(BiopbServicerBase):

    def __init__(self, model, tile_size: int = 1024, overlap_margin: int = 64):
        super().__init__()
        self.model = model
        # Lazy-path chunking. overlap_margin must exceed the largest expected
        # cell diameter for border IDs to stay consistent (see stitch.py); it is
        # additionally floored from the request diameter at run time.
        self.tile_size = tile_size
        self.overlap_margin = overlap_margin
        # Serializes ProcessImage GPU inference. _server_context's lock guards
        # RPC-vs-RPC, but an async lazy job runs in a background thread *after*
        # Run returns, so this lock is needed to keep it from racing a
        # concurrent (eager or lazy) inference on the single shared model.
        self._model_lock = threading.Lock()

    def RunDetection(self, request, context):
        with self._server_context(context):
            logger.info(f"Received message of size {request.ByteSize()}")

            image, kwargs = process_input(request)

            if image.ndim == 4:
                raise ValueError("3D input not supported. Use the 'ProcessImage' service instead.")

            logger.info(f"received image {image.shape}")

            preds = self.model.eval(image,  **kwargs,)

            response = process_result(preds, image)

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response


    def Run(self, request, context):
        with self._server_context(context):
            logger.info(f"Received message of size {request.ByteSize()}")

            image = get_image_data(request.image_data)

            # Start with default kwargs, merge request kwargs, validate.
            kwargs = parse_kwargs(request, _DEFAULT_KWARGS.copy())
            errors = validate_kwargs(kwargs, _CELLPOSE_kwargs_SCHEMA)
            if errors:
                raise ValueError("Invalid kwargs: " + "; ".join(errors))

            # Control parameter, not a model kwarg -- remove before model.eval.
            async_result = bool(kwargs.pop("async_result", False))

            # Lazy (dask) input may be larger than memory: process it tile-by-tile
            # and stream a lazy label-mask result back via the tensor cache.
            if isinstance(image, da.Array):
                return self._run_lazy(image, kwargs, async_result=async_result)

            image = ensure_eager(image)

            if image.shape[0] == 1:  # 2D
                image = image.squeeze(0)
            else:
                kwargs["do_3D"] = True

            logger.info(f"Decoded image {image.shape}")

            with self._model_lock:
                mask = self.model.eval(image, **kwargs)[0]

            response = proto.ProcessResponse(
                image_data=encode_image(mask),
            )

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response

    def _run_lazy(self, image, kwargs, async_result=False) -> proto.ProcessResponse:
        """Tile-wise ProcessImage for a lazy (dask) input, larger than memory.

        Mirrors the ucell lazy path: allocate the label-mask output at the
        co-located tensor cache, then run :func:`stitch.stitch_lazy_segmentation`
        over overlapping tiles. Each tile is fetched, run through the Cellpose
        network (flows only, ``compute_masks=False``) to get the spatial flow
        ``dP`` and ``cellprob``, integrated to per-pixel destinations, clustered
        with cross-tile ID inheritance, cropped to its core, and uploaded -- so
        neither the input nor the output is ever fully in memory.

        With ``async_result=False`` (default) the stitch runs inline and the
        returned ``lazy_data`` is a completed result. With ``async_result=True``
        the stitch runs on a background thread and the registration handle is
        returned immediately (avoiding a long-held RPC / client disconnect); the
        client must poll ``wait_for_upload_ready_pb`` before reading.
        """
        if self._tensor_cache is None:
            raise ValueError(
                "Lazy (dask) input requires a tensor cache to write the result. "
                "Start the server with --cache-dir."
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

        diameter = kwargs["diameter"]
        # Overlap must cover a whole cell so a straddling cell yields the same
        # destination in every tile that touches it (see stitch.py).
        margin = max(self.overlap_margin, int(np.ceil(2 * diameter)))
        core_shape = tuple(
            stitch.uniform_core(full_shape[i], self.tile_size) for i in range(2)
        )
        # Cellpose rescales each tile so mean diameter == 30 before the net and
        # integrates for (1/rescale)*200 == (diameter/30)*200 Euler steps.
        niter = max(1, int(round(diameter / 30.0 * 200)))
        logger.info(
            f"Lazy ProcessImage: image {image.shape}, core {core_shape}, "
            f"margin {margin}, niter {niter}"
        )

        from biopb.tensor import ChunkBounds

        template = da.zeros(full_shape, chunks=core_shape, dtype=np.int32)
        registration = self._tensor_cache.create_array(
            source_name=None, dim_labels=["Y", "X"], array_template=template
        )
        source_id = registration.tensor_descriptor.array_id

        def compute_chunk(tile_start, tile_stop):
            ys, xs = tile_start
            ye, xe = tile_stop
            if channel_axis is None:
                tile = image[ys:ye, xs:xe].compute()
            else:
                tile = image[ys:ye, xs:xe, :].compute()
            # Flows only -- skip Cellpose's own mask dynamics; we integrate and
            # stitch ourselves. flows[1] = dP [2,H,W], flows[2] = cellprob [H,W].
            # Lock guards the shared GPU model (a background async job may run
            # concurrently with another request's inference).
            with self._model_lock:
                _masks, flows, _styles = self.model.cp.eval(
                    tile,
                    channels=kwargs["channels"],
                    diameter=diameter,
                    normalize=kwargs["normalize"],
                    invert=kwargs["invert"],
                    cellprob_threshold=kwargs["cellprob_threshold"],
                    flow_threshold=0.0,
                    compute_masks=False,
                )
            dP = flows[1]
            cell_prob = flows[2]
            inds, p = dynamics_local.compute_destinations(
                dP,
                cell_prob,
                cellprob_threshold=kwargs["cellprob_threshold"],
                niter=niter,
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
                min_area=kwargs["min_size"],
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
                names=["cellpose"],
                op_schemas={
                    "cellpose": proto.OpSchema(
                        description="Cellpose Cyto3 cell segmentation model",
                        default_kwargs=default_kwargs,
                    ),
                }
            )


@app.command()
def main(
    modeltype: str = "cyto3",
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
    model = models.Cellpose(model_type=modeltype, gpu=gpu)

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
