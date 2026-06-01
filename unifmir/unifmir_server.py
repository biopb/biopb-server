"""gRPC servicer for the UNiFMIR image-restoration foundation model.

UNiFMIR is a restoration model (image -> image), so the natural RPC is
``ProcessImage.Run``. A single shared SwinIR backbone is specialized per task by
its own checkpoint ("head"); the head is selected per request via the
``op_name`` field (or the ``model_name`` kwarg), loaded lazily and cached. See
``heads.py`` for the registry and the per-task pre/post-processing.

Eager image data is processed in one shot. Lazy (dask) input -- larger than
memory -- is processed chunk-by-chunk with **no overlap** (the lazy backend does
not yet support halo reads), each chunk written once to a co-located tensor cache;
see ``tiling.py``. ``RunDetection`` is unimplemented (this is not a detector).
"""

import logging
import threading

import biopb.image as proto
import dask.array as da
import numpy as np
import typer

from biopb_image_base import (
    BiopbServicerBase,
    decode_image_data,
    encode_image,
    ensure_eager,
    parse_kwargs,
    run_server,
    validate_kwargs,
)

import heads as heads_mod
import tiling

app = typer.Typer(pretty_exceptions_enable=False)

logger = logging.getLogger(__name__)

_DEFAULT_KWARGS = {
    "model_name": "",       # selects the head; empty -> server default
    "async_result": False,  # lazy input: return the handle now, compute in background
}

_KWARGS_SCHEMA = {
    "model_name": {
        "type": "string",
        "description": "Head to run; one of " + ", ".join(heads_mod.HEADS),
    },
    "async_result": {
        "type": "bool",
        "description": "Lazy input only: return the result tensor handle "
                       "immediately and compute in the background.",
    },
}


def _prepare_image(image, spec: heads_mod.HeadSpec):
    """Coerce a decoded array (numpy or dask) to the head's expected spatial ndim.

    Drops leading singleton axes (e.g. a singleton Z or batch) and a trailing
    singleton channel, then validates the rank. Works on both eager (numpy) and
    lazy (dask) arrays since it only slices.
    """
    while image.ndim > spec.ndim and image.shape[0] == 1:
        image = image[0]
    if image.ndim == spec.ndim + 1 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim != spec.ndim:
        raise ValueError(
            f"Op '{spec.op_name}' expects a {spec.ndim}D image, "
            f"got array with shape {image.shape}."
        )
    return image


class UnifmrServicer(BiopbServicerBase):
    """Serves UNiFMIR restoration heads, selected per request."""

    def __init__(self, ckpt_dir: str, default_op: str, device, tile_size: int = 1024):
        # use_lock=False: we serialize only GPU inference (via _model_lock) so a
        # background lazy job and a foreground request can decode/encode in
        # parallel but never touch the model concurrently.
        super().__init__(use_lock=False)
        self.ckpt_dir = ckpt_dir
        self.default_op = default_op
        self.device = device
        self.tile_size = tile_size
        self._models: dict[str, object] = {}
        self._load_lock = threading.Lock()
        self._model_lock = threading.Lock()

    def _resolve_op(self, request, kwargs) -> heads_mod.HeadSpec:
        op = (getattr(request, "op_name", "") or kwargs["model_name"] or self.default_op)
        if op not in heads_mod.HEADS:
            raise ValueError(
                f"Unknown op '{op}'. Available: {', '.join(heads_mod.HEADS)}."
            )
        return heads_mod.HEADS[op]

    def _get_model(self, spec: heads_mod.HeadSpec):
        with self._load_lock:
            if spec.op_name not in self._models:
                self._models[spec.op_name] = heads_mod.load_head(
                    spec, self.ckpt_dir, self.device
                )
            return self._models[spec.op_name]

    def _infer(self, spec: heads_mod.HeadSpec, image: np.ndarray) -> np.ndarray:
        """Run one head on an in-memory (eager) array, serialized on the GPU."""
        model = self._get_model(spec)
        with self._model_lock:
            return heads_mod.predict(spec, image, model, self.device)

    def GetOpNames(self, request, context):
        with self._server_context(context):
            op_schemas = {}
            for name, spec in heads_mod.HEADS.items():
                singletons, multivalue = heads_mod.input_shape_hint(spec)
                op_schemas[name] = proto.OpSchema(
                    description=spec.description,
                    labels=list(spec.labels),
                    input_shape_hint=proto.InputShapeHint(
                        expected_singletons=singletons,
                        required_multivalue=multivalue,
                    ),
                )
            return proto.OpNames(names=list(heads_mod.HEADS), op_schemas=op_schemas)

    def Run(self, request, context):
        with self._server_context(context):
            kwargs = parse_kwargs(request, _DEFAULT_KWARGS)
            errors = validate_kwargs(kwargs, _KWARGS_SCHEMA)
            if errors:
                raise ValueError("Invalid kwargs: " + "; ".join(errors))

            spec = self._resolve_op(request, kwargs)
            image = decode_image_data(request.image_data)

            if isinstance(image, da.Array):
                return self._run_lazy(spec, image, async_result=bool(kwargs["async_result"]))

            image = ensure_eager(image)
            image = _prepare_image(image, spec)
            logger.info("Run op=%s image=%s (eager)", spec.op_name, image.shape)
            result = self._infer(spec, image)
            logger.info("op=%s produced output %s", spec.op_name, result.shape)
            return proto.ProcessResponse(image_data=encode_image(result))

    def _run_lazy(self, spec, image, async_result=False) -> proto.ProcessResponse:
        """Chunk-wise ProcessImage for a lazy (dask) input, larger than memory.

        Each Y/X chunk is fetched, run through the head independently, and
        uploaded to a co-located tensor cache -- no overlap, no blending, every
        output pixel written exactly once (``tiling.tile_plane``). With
        ``async_result`` the work runs on a background thread and the handle is
        returned immediately; otherwise it runs inline and the completed result
        is returned.
        """
        if self._tensor_cache is None:
            raise ValueError(
                "Lazy (dask) input requires a tensor cache to write the result. "
                "Start the server with --cache-dir."
            )

        image = _prepare_image(image, spec)
        up = spec.upscale

        if spec.ndim == 2:
            height, width = int(image.shape[0]), int(image.shape[1])
            zdim = None
            out_shape = (height * up, width * up)
            out_dims = ["Y", "X"]
        else:  # 3D Z-stack: chunk the Y/X plane, keep full Z per chunk
            zdim, height, width = (int(image.shape[0]), int(image.shape[1]), int(image.shape[2]))
            out_shape = (zdim, height * up, width * up)
            out_dims = ["Z", "Y", "X"]

        core = tiling.plane_core_shape((height, width), self.tile_size)
        out_core = (core[0] * up, core[1] * up)
        chunks = out_core if zdim is None else (zdim, out_core[0], out_core[1])

        template = da.zeros(out_shape, chunks=chunks, dtype=np.float32)
        registration = self._tensor_cache.create_array(
            source_name=None, dim_labels=out_dims, array_template=template
        )
        source_id = registration.tensor_descriptor.array_id
        logger.info(
            "Lazy op=%s image=%s core=%s upscale=%d -> out=%s",
            spec.op_name, image.shape, core, up, out_shape,
        )

        from biopb.tensor import ChunkBounds

        def compute_chunk(y0, y1, x0, x1):
            tile = (image[y0:y1, x0:x1] if zdim is None
                    else image[:, y0:y1, x0:x1]).compute()
            return self._infer(spec, tile)

        def write_chunk(oy0, oy1, ox0, ox1, data):
            start = [oy0, ox0] if zdim is None else [0, oy0, ox0]
            stop = [oy1, ox1] if zdim is None else [zdim, oy1, ox1]
            self._tensor_cache.upload_array_chunks(
                source_id, ChunkBounds(start=start, stop=stop),
                np.ascontiguousarray(data, dtype=np.float32),
            )

        def run():
            n = tiling.tile_plane((height, width), core, up, compute_chunk, write_chunk)
            logger.info("Lazy op=%s wrote %d chunks for source %s", spec.op_name, n, source_id)

        if async_result:
            def worker():
                try:
                    run()
                except Exception:
                    logger.exception(
                        "[async] lazy op=%s FAILED for source %s; client poll will time out",
                        spec.op_name, source_id,
                    )

            threading.Thread(target=worker, name=f"lazy-{source_id}").start()
            logger.info("Lazy op=%s (async): returning handle for source %s", spec.op_name, source_id)
            return proto.ProcessResponse(image_data=proto.ImageData(lazy_data=registration))

        run()
        serialized = self._tensor_cache.to_serialized_tensor(source_id)
        return proto.ProcessResponse(image_data=proto.ImageData(lazy_data=serialized))

    def RunDetection(self, request, context):
        with self._server_context(context):
            raise NotImplementedError(
                "UNiFMIR is an image-restoration model; use the ProcessImage.Run "
                "endpoint, not ObjectDetection."
            )


@app.command()
def main(
    ckpt_dir: str = typer.Option("./experiment", help="Directory holding the UNiFMIR checkpoints."),
    default_op: str = typer.Option("sr_factin", help="Head used when a request omits op_name/model_name."),
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
        1024, help="Target (non-overlap) chunk size for lazy processing."),
):
    import torch

    if default_op not in heads_mod.HEADS:
        raise typer.BadParameter(
            f"--default-op '{default_op}' is not a known head: {', '.join(heads_mod.HEADS)}"
        )

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    logger.info("Using device %s", device)

    run_server(
        UnifmrServicer(ckpt_dir=ckpt_dir, default_op=default_op, device=device, tile_size=tile_size),
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
