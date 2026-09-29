"""Cellpose-SAM cell segmentation, served over the biopb.image Ops protocol.

One op, ``cellpose``. An image up to ``_WHOLE_IMAGE_MAX`` bytes is segmented
whole, in 2D or, with a non-singleton Z, in 3D. A larger 2D image is read and
segmented in overlapping tiles whose labels are stitched into one consistent
label image (``biopb_image_base.stitch``); the input is never held whole.

    python cellpose_server.py --host 127.0.0.1 --port 50051 [--cache-dir DIR]
"""

import logging
import tempfile
import threading
from functools import lru_cache

import numpy as np
from biopb_image_base import Tensor, dynamics_local, op, serve, stitch

logger = logging.getLogger(__name__)

# Above this many input bytes a 2D image is segmented tile by tile.
_WHOLE_IMAGE_MAX = 256 * 1024**2

# Tiling. The overlap must exceed the largest cell so a cell straddling a
# border yields the same destination in every tile that sees it; it is also
# floored at twice the diameter per call.
_TILE_SIZE = 1024
_OVERLAP_MARGIN = 64

_NITER = 200  # cellpose's default Euler-integration steps (eval niter=None)

# The network is not thread safe, and the server runs ops on a thread pool.
_model_lock = threading.Lock()


@lru_cache(maxsize=1)
def _model():
    from cellpose import io, models

    io.logger_setup()
    return models.CellposeModel(gpu=True)


def _check(diameter, cellprob_threshold, min_area):
    if not 0.0 <= diameter <= 1000.0:
        raise ValueError(f"diameter must be in [0, 1000], got {diameter}")
    if not -6.0 <= cellprob_threshold <= 6.0:
        raise ValueError(
            f"cellprob_threshold must be in [-6, 6], got {cellprob_threshold}"
        )
    if min_area < 0:
        raise ValueError(f"min_area must be >= 0, got {min_area}")


def _eval_kwargs(diameter, cellprob_threshold):
    """cellpose ``eval`` kwargs; a diameter of 0 is left to cellpose to estimate."""
    kwargs = {"cellprob_threshold": cellprob_threshold}
    if diameter:
        kwargs["diameter"] = diameter
    return kwargs


@op(
    description="Cellpose-SAM cell segmentation",
    labels=["segmentation"],
    input="lazy",
)
def cellpose(
    image: Tensor("ZYXC"),
    diameter: float = 0.0,
    cellprob_threshold: float = 0.0,
    min_area: int = 15,
):
    """A label image of the cells: 2D, or 3D when the image has a Z extent.

    diameter is the cell diameter in pixels; 0 estimates it.
    """
    _check(diameter, cellprob_threshold, min_area)
    is_3d = image.shape[0] > 1
    eval_kwargs = _eval_kwargs(diameter, cellprob_threshold)

    if image.nbytes <= _WHOLE_IMAGE_MAX:
        array = np.asarray(image.compute())
        with _model_lock:
            if is_3d:
                masks = _model().eval(
                    array,
                    channel_axis=-1,
                    z_axis=0,
                    do_3D=True,
                    flow3D_smooth=1,
                    min_size=min_area,
                    **eval_kwargs,
                )[0]
            else:
                masks = _model().eval(array[0], min_size=min_area, **eval_kwargs)[0]
        masks = masks.astype(np.uint32)
        return (masks if is_3d else masks[None])[..., None]

    if is_3d:
        raise ValueError(
            f"a 3D image is segmented whole, and this one is {image.nbytes} bytes, "
            f"more than {_WHOLE_IMAGE_MAX}; send a 2D plane or a smaller volume"
        )
    return _tiled(image[0], diameter, cellprob_threshold, min_area)[None, ..., None]


def _tiled(image, diameter, cellprob_threshold, min_area):
    """Segment a lazy YXC image in overlapping tiles into a disk-backed label
    image.

    Each tile runs through the network for flows only; the flows are integrated
    to per-pixel destinations and clustered with cross-tile ID inheritance, and
    each tile's core is written to the output.
    """
    full_shape = tuple(image.shape[:2])
    margin = max(_OVERLAP_MARGIN, int(np.ceil(2 * diameter)))
    core_shape = tuple(stitch.uniform_core(n, _TILE_SIZE) for n in full_shape)
    logger.info(
        "tiled cellpose-sam: image %s, core %s, margin %d",
        image.shape, core_shape, margin,
    )

    # An unlinked temporary file: the mapping keeps it alive until the result
    # has been sent.
    labels = np.memmap(tempfile.TemporaryFile(), dtype=np.uint32, mode="w+",
                       shape=full_shape)
    eval_kwargs = _eval_kwargs(diameter, cellprob_threshold)

    def compute_chunk(tile_start, tile_stop):
        (ys, xs), (ye, xe) = tile_start, tile_stop
        tile = np.asarray(image[ys:ye, xs:xe].compute())
        with _model_lock:
            # flows[1] is dP (2, H, W), flows[2] the cell probability (H, W).
            _masks, flows, _styles = _model().eval(
                tile, compute_masks=False, **eval_kwargs
            )
        # Cellpose's dP already has the scale compute_destinations expects.
        return dynamics_local.compute_destinations(
            flows[1], flows[2], cellprob_threshold=cellprob_threshold, niter=_NITER
        )

    def write_core(core_start, core_stop, core_labels):
        (ys, xs), (ye, xe) = core_start, core_stop
        labels[ys:ye, xs:xe] = core_labels

    n = stitch.stitch_lazy_segmentation(
        full_shape, core_shape, margin, compute_chunk, write_core, min_area=min_area
    )
    logger.info("tiled cellpose-sam: %d cells", n)
    return labels


if __name__ == "__main__":
    serve()
