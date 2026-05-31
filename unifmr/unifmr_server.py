"""gRPC servicer for the UNiFMIR image-restoration foundation model.

UNiFMIR is a restoration model (image -> image), so the natural RPC is
``ProcessImage.Run``. A single shared SwinIR backbone is specialized per task by
its own checkpoint ("head"); the head is selected per request via the
``op_name`` field (or the ``model_name`` kwarg), loaded lazily and cached. See
``heads.py`` for the registry and the per-task pre/post-processing.

Phase 1 handles eager image data only. ``RunDetection`` is unimplemented (this is
not a detector).
"""

import logging
import threading

import biopb.image as proto
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

app = typer.Typer(pretty_exceptions_enable=False)

logger = logging.getLogger(__name__)

_DEFAULT_KWARGS = {
    "model_name": "",  # selects the head; empty -> server default
}

_KWARGS_SCHEMA = {
    "model_name": {
        "type": "string",
        "description": "Head to run; one of " + ", ".join(heads_mod.HEADS),
    },
}


def _prepare_image(image: np.ndarray, spec: heads_mod.HeadSpec) -> np.ndarray:
    """Coerce a decoded array to the spatial ndim the head expects.

    Drops leading singleton axes (e.g. a singleton Z or batch) and a trailing
    singleton channel, then validates the rank.
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

    def __init__(self, ckpt_dir: str, default_op: str, device):
        super().__init__()  # use_lock=True: serialize (torch model is not reentrant)
        self.ckpt_dir = ckpt_dir
        self.default_op = default_op
        self.device = device
        self._models: dict[str, object] = {}
        self._load_lock = threading.Lock()

    def _resolve_op(self, request) -> heads_mod.HeadSpec:
        kwargs = parse_kwargs(request, _DEFAULT_KWARGS)
        errors = validate_kwargs(kwargs, _KWARGS_SCHEMA)
        if errors:
            raise ValueError("Invalid kwargs: " + "; ".join(errors))

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
            spec = self._resolve_op(request)

            image = decode_image_data(request.image_data)
            image = ensure_eager(image)
            image = _prepare_image(image, spec)

            logger.info("Run op=%s image=%s", spec.op_name, image.shape)

            model = self._get_model(spec)
            result = heads_mod.predict(spec, image, model, self.device)

            logger.info("op=%s produced output %s", spec.op_name, result.shape)

            return proto.ProcessResponse(image_data=encode_image(result))

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
):
    import torch

    if default_op not in heads_mod.HEADS:
        raise typer.BadParameter(
            f"--default-op '{default_op}' is not a known head: {', '.join(heads_mod.HEADS)}"
        )

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    logger.info("Using device %s", device)

    run_server(
        UnifmrServicer(ckpt_dir=ckpt_dir, default_op=default_op, device=device),
        port=port,
        workers=workers,
        ip=ip,
        local=local,
        token=token,
        log_level="DEBUG" if debug else "INFO",
        compression=compression,
    )


if __name__ == "__main__":
    app()
