"""SAMCell cell segmentation, served over the biopb.image Ops protocol.

One op, ``samcell``: a 2D image is segmented by a sliding window over the
whole image, so the image is read whole (``input="eager"``).

    python samcell_server.py --host 127.0.0.1 --port 50051
"""

import logging
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from biopb_image_base import Tensor, op, serve

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent / "samcell-generalist.pt"

# The pipeline is not thread safe, and the server runs ops on a thread pool.
_pipeline_lock = threading.Lock()


@lru_cache(maxsize=1)
def _pipeline():
    import torch
    from model import FinetunedSAM
    from pipeline import SlidingWindowPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("No GPU available. This might be very slow...")
    model = FinetunedSAM(
        "facebook/sam-vit-base",
        finetune_vision=False,
        finetune_prompt=True,
        finetune_decoder=True,
    )
    model.load_weights(_MODEL_PATH)
    return SlidingWindowPipeline(model, device, crop_size=256)


@op(
    description="SAMCell cell segmentation",
    labels=["segmentation"],
)
def samcell(image: Tensor("YXC")):
    """A label image of the cells. The image must be 2D: YX or YXC.

    Multiple channels are averaged to grayscale.
    """
    gray = np.asarray(image, dtype=np.float32).mean(axis=-1)
    span = gray.max() - gray.min()
    if span == 0:
        return np.zeros(gray.shape + (1,), dtype=np.uint32)
    gray = ((gray - gray.min()) / span * 255).astype("uint8")

    with _pipeline_lock:
        masks = _pipeline().run(gray)
    logger.info("samcell: %d cells in %s", masks.max(), gray.shape)
    return masks.astype(np.uint32)[..., None]


if __name__ == "__main__":
    serve()
