"""Cellpose (cyto3) cell segmentation, served over the biopb.image Ops protocol.

One op, ``cellpose``. An image up to ``_WHOLE_IMAGE_MAX`` bytes is segmented
whole, in 2D or, with a non-singleton Z, in 3D. A larger 2D image is read and
segmented in overlapping tiles whose labels are stitched into one consistent
label image (``biopb_image_base.stitch``); the input is never held whole.

    python cellpose_server.py --host 127.0.0.1 --port 50051 [--cache-dir DIR]
"""

import logging
import tempfile
from functools import lru_cache

import numpy as np
from biopb_image_base import Tensor, dynamics_local, op, serve, stitch

logger = logging.getLogger(__name__)

_MODEL_TYPE = "cyto3"

# Above this many input bytes a 2D image is segmented tile by tile.
_WHOLE_IMAGE_MAX = 256 * 1024**2

# Tiling. The overlap must exceed the largest cell so a cell straddling a
# border yields the same destination in every tile that sees it; it is also
# floored at twice the diameter per call.
_TILE_SIZE = 1024
_OVERLAP_MARGIN = 64


@lru_cache(maxsize=1)
def _model():
    from cellpose import models

    return models.Cellpose(model_type=_MODEL_TYPE, gpu=True)


def _check(channels, diameter, flow_threshold, cellprob_threshold, min_size):
    if len(channels) != 2:
        raise ValueError(f"channels is [cytoplasm, nucleus], got {channels!r}")
    if diameter < 0:
        raise ValueError(f"diameter must be >= 0, got {diameter}")
    if not 0.0 <= flow_threshold <= 1.0:
        raise ValueError(f"flow_threshold must be in [0, 1], got {flow_threshold}")
    if not -6.0 <= cellprob_threshold <= 6.0:
        raise ValueError(
            f"cellprob_threshold must be in [-6, 6], got {cellprob_threshold}"
        )
    if min_size < 0:
        raise ValueError(f"min_size must be >= 0, got {min_size}")


@op(
    description="Cellpose cyto3 cell segmentation",
    labels=["segmentation"],
    input="lazy",
)
def cellpose(
    image: Tensor("ZYXC"),
    channels: list = (0, 0),
    diameter: float = 30.0,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    normalize: bool = True,
    invert: bool = False,
    min_size: int = 15,
):
    """A label image of the cells: 2D, or 3D when the image has a Z extent.

    channels is cellpose's [cytoplasm, nucleus], 1-based, 0 for grayscale.
    """
    channels = [int(c) for c in channels]
    _check(channels, diameter, flow_threshold, cellprob_threshold, min_size)
    is_3d = image.shape[0] > 1

    if image.nbytes <= _WHOLE_IMAGE_MAX:
        array = np.asarray(image.compute())
        masks = _model().eval(
            array if is_3d else array[0],
            channels=channels,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            normalize=normalize,
            invert=invert,
            min_size=min_size,
            do_3D=is_3d,
        )[0]
        masks = masks.astype(np.uint32)
        return (masks if is_3d else masks[None])[..., None]

    if is_3d:
        raise ValueError(
            f"a 3D image is segmented whole, and this one is {image.nbytes} bytes, "
            f"more than {_WHOLE_IMAGE_MAX}; send a 2D plane or a smaller volume"
        )
    return _tiled(
        image[0],
        channels=channels,
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        normalize=normalize,
        invert=invert,
        min_size=min_size,
    )[None, ..., None]


def _tiled(image, *, channels, diameter, cellprob_threshold, normalize, invert, min_size):
    """Segment a lazy YXC image in overlapping tiles into a disk-backed label
    image.

    Each tile runs through the network for flows only; the flows are integrated
    to per-pixel destinations and clustered with cross-tile ID inheritance, and
    each tile's core is written to the output.
    """
    full_shape = tuple(image.shape[:2])
    diameter = diameter or 30.0
    margin = max(_OVERLAP_MARGIN, int(np.ceil(2 * diameter)))
    core_shape = tuple(stitch.uniform_core(n, _TILE_SIZE) for n in full_shape)
    # Cellpose rescales a tile so the mean diameter is 30 and integrates
    # (diameter / 30) * 200 Euler steps.
    niter = max(1, int(round(diameter / 30.0 * 200)))
    logger.info(
        "tiled cellpose: image %s, core %s, margin %d, niter %d",
        image.shape, core_shape, margin, niter,
    )

    # An unlinked temporary file: the mapping keeps it alive until the result
    # has been sent.
    labels = np.memmap(tempfile.TemporaryFile(), dtype=np.uint32, mode="w+",
                       shape=full_shape)
    net = _model().cp

    def compute_chunk(tile_start, tile_stop):
        (ys, xs), (ye, xe) = tile_start, tile_stop
        tile = np.asarray(image[ys:ye, xs:xe].compute())
        if tile.shape[-1] == 1:
            tile = tile[..., 0]
        # flows[1] is dP (2, H, W), flows[2] the cell probability (H, W).
        _masks, flows, _styles = net.eval(
            tile,
            channels=channels,
            diameter=diameter,
            normalize=normalize,
            invert=invert,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=0.0,
            compute_masks=False,
        )
        return dynamics_local.compute_destinations(
            flows[1], flows[2], cellprob_threshold=cellprob_threshold, niter=niter
        )

    def write_core(core_start, core_stop, core_labels):
        (ys, xs), (ye, xe) = core_start, core_stop
        labels[ys:ye, xs:xe] = core_labels

    n = stitch.stitch_lazy_segmentation(
        full_shape, core_shape, margin, compute_chunk, write_core, min_area=min_size
    )
    logger.info("tiled cellpose: %d cells", n)
    return labels


if __name__ == "__main__":
    serve()
