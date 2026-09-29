"""UCell (FRM-based) cell segmentation, served over the biopb.image Ops protocol.

One op, ``ucell``. An image up to ``_WHOLE_IMAGE_MAX`` bytes is segmented whole.
A larger image is read and segmented in overlapping tiles whose labels are
stitched into one consistent label image (``biopb_image_base.stitch``); the
input is never held whole.

    python ucell_server.py --host 127.0.0.1 --port 50051 [--cache-dir DIR]
"""

import logging
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from biopb_image_base import Tensor, dynamics_local, op, serve, stitch

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_MODEL_PATH = _HERE / "model.pt"
_SMALL_MODEL_PATH = _HERE / "model_s.pt"

# Above this many input bytes an image is segmented tile by tile.
_WHOLE_IMAGE_MAX = 256 * 1024**2

# Tiling. The overlap must exceed the largest cell so a cell straddling a
# border yields the same destination in every tile that sees it.
_TILE_SIZE = 1024
_OVERLAP_MARGIN = 64

_NITER = 500  # Integration steps for mask computation

# The network is not thread safe, and the server runs ops on a thread pool.
_model_lock = threading.Lock()


@lru_cache(maxsize=2)
def _model(small_model: bool):
    """The (network, config, device) for the large or the small checkpoint."""
    import torch
    from ucell.frm import FRMWrapper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("No GPU available. This might be very slow...")

    # The checkpoint carries its config. Inference recurses once, not as deep as
    # training did.
    wrapper = FRMWrapper.from_checkpoint(
        _SMALL_MODEL_PATH if small_model else _MODEL_PATH,
        overrides={"halt_max_steps": 1},
    ).eval()
    net = wrapper.inner.to(device)
    if device.type == "cuda":
        net = torch.compile(net)
    logger.info("loaded %s model on %s", "small" if small_model else "large", device)
    return net, wrapper.config, device


def _format(img: np.ndarray) -> np.ndarray:
    """Normalize and pad the channels of a YX or YXC image for the network."""
    from ucell.utils import pad_channel

    img = img / (img.max() + 1e-5)
    return pad_channel(img)


def _infer(img: np.ndarray, task_id: int, small_model: bool):
    """The flow (2, H, W) and cell probability (H, W) of a YX or YXC image."""
    import torch
    from ucell.utils import patcherize

    net, config, device = _model(small_model)
    predict = patcherize(net.predict, GS=config.image_size)
    with _model_lock, torch.device(device):
        output = predict(_format(img), task_id)
    # output is (H, W, 3): flow[:2], cell_prob[2]
    return np.moveaxis(output[:, :, :2], -1, 0), output[:, :, 2]


def _check(task_id, cellprob_threshold, min_area):
    if task_id < 0:
        raise ValueError(f"task_id must be >= 0, got {task_id}")
    if not -6.0 <= cellprob_threshold <= 6.0:
        raise ValueError(
            f"cellprob_threshold must be in [-6, 6], got {cellprob_threshold}"
        )
    if min_area < 0:
        raise ValueError(f"min_area must be >= 0, got {min_area}")


@op(
    description="UCell cell segmentation model (FRM-based)",
    labels=["segmentation"],
    input="lazy",
)
def ucell(
    image: Tensor("YXC"),
    task_id: int = 0,
    cellprob_threshold: float = -0.2,
    min_area: int = 5,
    small_model: bool = False,
):
    """A label image of the cells. The image must be 2D: YX or YXC.

    task_id selects the task of the multi-task model. small_model uses the
    768-wide checkpoint instead of the 1024-wide one.
    """
    _check(task_id, cellprob_threshold, min_area)
    if image.shape[-1] > 3:
        raise ValueError(f"ucell takes 1 to 3 channels, got {image.shape[-1]}")
    if image.nbytes <= _WHOLE_IMAGE_MAX:
        from ucell.dynamics import compute_masks

        _net, _config, device = _model(small_model)
        array = np.asarray(image.compute())
        flow, cell_prob = _infer(array, task_id, small_model)
        masks = compute_masks(
            flow * 4.0,
            cell_prob,
            niter=_NITER,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=0,
            min_size=min_area,
            max_size_fraction=0.4,
            device=device,
        )
        return masks.astype(np.uint32)[..., None]

    return _tiled(image, task_id, cellprob_threshold, min_area, small_model)[
        ..., None
    ]


def _tiled(image, task_id, cellprob_threshold, min_area, small_model):
    """Segment a lazy YXC image in overlapping tiles into a disk-backed label
    image.

    Each tile runs through the network for flows only; the flows are integrated
    to per-pixel destinations and clustered with cross-tile ID inheritance, and
    each tile's core is written to the output.
    """
    full_shape = tuple(image.shape[:2])
    core_shape = tuple(stitch.uniform_core(n, _TILE_SIZE) for n in full_shape)
    logger.info(
        "tiled ucell: image %s, core %s, margin %d",
        image.shape, core_shape, _OVERLAP_MARGIN,
    )

    # An unlinked temporary file: the mapping keeps it alive until the result
    # has been sent.
    labels = np.memmap(tempfile.TemporaryFile(), dtype=np.uint32, mode="w+",
                       shape=full_shape)

    def compute_chunk(tile_start, tile_stop):
        (ys, xs), (ye, xe) = tile_start, tile_stop
        tile = np.asarray(image[ys:ye, xs:xe].compute())
        flow, cell_prob = _infer(tile, task_id, small_model)
        # Mirror the whole-image path: dP = flow * 4.0, flow_threshold off.
        return dynamics_local.compute_destinations(
            flow * 4.0, cell_prob,
            cellprob_threshold=cellprob_threshold, niter=_NITER,
        )

    def write_core(core_start, core_stop, core_labels):
        (ys, xs), (ye, xe) = core_start, core_stop
        labels[ys:ye, xs:xe] = core_labels

    n = stitch.stitch_lazy_segmentation(
        full_shape, core_shape, _OVERLAP_MARGIN, compute_chunk, write_core,
        min_area=min_area,
    )
    logger.info("tiled ucell: %d cells", n)
    return labels


if __name__ == "__main__":
    serve()
