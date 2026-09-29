"""UNiFMIR image restoration, served over the biopb.image Ops protocol.

UNiFMIR is a restoration model (image -> image). A single shared SwinIR backbone
is specialized per task by its own checkpoint (a "head"), and each head is an op
of its own, named as in ``heads.HEADS``: ``sr_*`` take a 2D image (YX) and return
one at twice the size, ``denoise_*`` and ``isotropic_*`` take a Z-stack (ZYX) and
return one of the same shape. A head is loaded on first use. See ``heads.py`` for
the registry and the per-task pre/post-processing.

An image up to ``_WHOLE_IMAGE_MAX`` bytes is restored whole. A larger one is read
and restored in chunks of the Y/X plane with **no overlap** (seams between chunks
are accepted); the input is never held whole. See ``tiling.py``.

    python unifmir_server.py --host 127.0.0.1 --port 50051 [--cache-dir DIR]
"""

import logging
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from biopb_image_base import Tensor, op, serve

import heads as heads_mod
import tiling

logger = logging.getLogger(__name__)

# Where the checkpoint folders are baked into the image.
_CKPT_DIR = Path(__file__).parent / "experiment"

# Above this many input bytes an image is restored chunk by chunk.
_WHOLE_IMAGE_MAX = 256 * 1024**2

# Target (non-overlap) chunk size for the Y/X plane.
_TILE_SIZE = 1024

# The heads share one GPU; serialize inference on it.
_model_lock = threading.Lock()


def _device():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("No GPU available. This might be very slow...")
    return device


@lru_cache(maxsize=None)
def _head(op_name: str):
    """The (network, device) of a head, loaded on first use."""
    device = _device()
    return heads_mod.load_head(heads_mod.HEADS[op_name], str(_CKPT_DIR), device), device


def _infer(spec: heads_mod.HeadSpec, image: np.ndarray) -> np.ndarray:
    """Run one head on an in-memory array, serialized on the GPU."""
    with _model_lock:
        model, device = _head(spec.op_name)
        return heads_mod.predict(spec, image, model, device)


def _restore(spec: heads_mod.HeadSpec, image):
    """The head's output for an image of its own rank, whole or chunk by chunk."""
    if image.nbytes <= _WHOLE_IMAGE_MAX:
        result = _infer(spec, np.asarray(image.compute()))
        logger.info("%s: %s -> %s", spec.op_name, image.shape, result.shape)
        return result
    return _tiled(spec, image)


def _tiled(spec: heads_mod.HeadSpec, image):
    """Restore a lazy YX or ZYX image in non-overlapping chunks of the Y/X plane
    into a disk-backed result. A Z-stack keeps its whole Z extent in every chunk.
    """
    up = spec.upscale
    stack = spec.ndim == 3
    zdim = int(image.shape[0]) if stack else None
    height, width = int(image.shape[-2]), int(image.shape[-1])
    core = tiling.plane_core_shape((height, width), _TILE_SIZE)
    out_shape = ((zdim,) if stack else ()) + (height * up, width * up)
    logger.info("tiled %s: image %s, core %s, out %s", spec.op_name, image.shape, core, out_shape)

    # An unlinked temporary file: the mapping keeps it alive until the result
    # has been sent.
    out = np.memmap(tempfile.TemporaryFile(), dtype=np.float32, mode="w+", shape=out_shape)

    def compute_chunk(y0, y1, x0, x1):
        tile = image[:, y0:y1, x0:x1] if stack else image[y0:y1, x0:x1]
        return _infer(spec, np.asarray(tile.compute()))

    def write_chunk(oy0, oy1, ox0, ox1, data):
        out[..., oy0:oy1, ox0:ox1] = data

    n = tiling.tile_plane((height, width), core, up, compute_chunk, write_chunk)
    logger.info("tiled %s: %d chunks", spec.op_name, n)
    return out


def _declare(spec: heads_mod.HeadSpec):
    """Declare the op for one head."""

    def run(image):
        return _restore(spec, image).astype(np.float32, copy=False)

    run.__name__ = run.__qualname__ = spec.op_name
    run.__doc__ = spec.description
    run.__annotations__ = {"image": Tensor("YX" if spec.ndim == 2 else "ZYX")}
    return op(
        name=spec.op_name,
        description=spec.description,
        labels=list(spec.labels),
        input="lazy",
    )(run)


for _spec in heads_mod.HEADS.values():
    _declare(_spec)


if __name__ == "__main__":
    serve()
