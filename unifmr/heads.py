"""UNiFMIR task-head registry and inference runners.

Each *head* is the shared SwinIR backbone specialized for one restoration task
by its own checkpoint. Phase 1 exposes only the 2D image->image families:

  * Super-resolution (``sr_*``)  -- 2D in, 2x 2D out.
  * Denoising (``denoise_*``)    -- 3D Z-stack, processed slice-wise with a
                                    window of neighbouring slices.
  * Isotropic reconstruction     -- 3D Z-stack, two rotated passes combined by
    (``isotropic_*``)             geometric mean.

The model and the per-task pre/post-processing are ported faithfully from the
upstream ``app.py`` (the gradio demo), but the csbdeep/TensorFlow normalization
helpers are reimplemented inline here so the service only needs torch/timm/einops.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from model.swinir import swinir as SwinIR

logger = logging.getLogger(__name__)

_EPS = 1e-20
# Batch size (number of rotated columns/rows) per model call in the isotropic
# runner; purely a throughput/memory knob, does not affect results.
_ISO_BATCH = 8


# --------------------------------------------------------------------------- #
# Normalization helpers (reimplemented from csbdeep to avoid the TF dependency)
# --------------------------------------------------------------------------- #
def _minmax_normalize(x: np.ndarray) -> np.ndarray:
    """csbdeep.normalize(x, pmin=0, pmax=100, clip=True): scale [min,max]->[0,1]."""
    mi = float(x.min())
    ma = float(x.max())
    return np.clip((x - mi) / (ma - mi + _EPS), 0.0, 1.0).astype(np.float32)


def _percentile_normalize(
    x: np.ndarray, pmin: float = 2.0, pmax: float = 99.8
) -> tuple[np.ndarray, float, float]:
    """PercentileNormalizer.before over the whole volume; returns (norm, mi, ma)."""
    mi = float(np.percentile(x, pmin))
    ma = float(np.percentile(x, pmax))
    norm = ((x - mi) / (ma - mi + _EPS)).astype(np.float32)
    return norm, mi, ma


def _percentile_denormalize(x: np.ndarray, mi: float, ma: float) -> np.ndarray:
    """PercentileNormalizer.after: alpha=ma-mi, beta=mi -> alpha*x + beta."""
    return (x * (ma - mi) + mi).astype(np.float32)


def _rotate(arr: np.ndarray, k: int = 1, axis: int = 1, copy: bool = True) -> np.ndarray:
    """Rotate by 90 degrees around the first two axes (verbatim from app.py)."""
    if copy:
        arr = arr.copy()
    k = k % 4
    arr = np.rollaxis(arr, axis, arr.ndim)
    if k == 0:
        res = arr
    elif k == 1:
        res = arr[::-1].swapaxes(0, 1)
    elif k == 2:
        res = arr[::-1, ::-1]
    else:
        res = arr.swapaxes(0, 1)[::-1]
    res = np.rollaxis(res, -1, axis)
    return res


def _denoise_window_indices(ch: int, depth: int, in_chans: int) -> list[int]:
    """Z indices of the neighbour window centred on slice ``ch`` (matches app.py)."""
    half = in_chans // 2
    if in_chans == 1:
        return [ch]
    if ch < half:
        return [ch] * (half - ch) + list(range(0, half + ch + 1))
    if ch >= depth - half:
        numa = (half - (depth - ch)) + 1
        return list(range(ch - half, depth)) + [ch] * numa
    return list(range(ch - half, ch + half + 1))


# --------------------------------------------------------------------------- #
# Inference runners. Each takes the decoded image (numpy, channel-less), the
# loaded torch module, the device and the HeadSpec; returns a numpy result.
# --------------------------------------------------------------------------- #
def _predict_sr(image: np.ndarray, model, device, spec: "HeadSpec") -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"SR expects a 2D image, got shape {image.shape}")
    x = _minmax_normalize(image.astype(np.float32))
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        y = model(t)
    y = torch.clamp(y, 0.0, 1.0)  # utility.quantize(rgb_range=1)
    return y.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def _predict_denoise(image: np.ndarray, model, device, spec: "HeadSpec") -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Denoising expects a 3D Z-stack, got shape {image.shape}")
    norm, mi, ma = _percentile_normalize(image.astype(np.float32))
    t = torch.from_numpy(norm).to(device)  # (Z, H, W)
    depth = t.shape[0]
    out = torch.zeros_like(t)
    with torch.no_grad():
        for ch in range(depth):
            idx = _denoise_window_indices(ch, depth, spec.in_chans)
            window = t[idx].unsqueeze(0)  # (1, in_chans, H, W)
            out[ch] = model(window)[0, 0]
    return _percentile_denormalize(out.cpu().numpy(), mi, ma)


def _predict_isotropic(image: np.ndarray, model, device, spec: "HeadSpec") -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Isotropic expects a 3D Z-stack, got shape {image.shape}")
    norm, mi, ma = _percentile_normalize(image.astype(np.float32))
    lr = np.expand_dims(norm, -1)  # (D, H, W, 1)
    isoim1 = np.zeros_like(lr, dtype=np.float32)
    isoim2 = np.zeros_like(lr, dtype=np.float32)

    def _run(batch_np: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(batch_np)).float().to(device)
        with torch.no_grad():
            return model(t).cpu().numpy()

    # Pass 1: resolve along the width axis.
    width = lr.shape[2]
    for w0 in range(0, width, _ISO_BATCH):
        w1 = min(w0 + _ISO_BATCH, width)
        x_rot1 = _rotate(lr[:, :, w0:w1, :], axis=1, copy=False)  # (Wb, H, D, 1)
        x_rot1 = np.expand_dims(np.squeeze(x_rot1, -1), 1)        # (Wb, 1, H, D)
        a1 = _run(x_rot1)                                          # (Wb, 1, H, D)
        a1 = np.expand_dims(np.squeeze(a1, 1), -1)               # (Wb, H, D, 1)
        isoim1[:, :, w0:w1, :] = _rotate(a1, -1, axis=1, copy=False)

    # Pass 2: resolve along the height axis.
    height = lr.shape[1]
    for h0 in range(0, height, _ISO_BATCH):
        h1 = min(h0 + _ISO_BATCH, height)
        x_rot2 = _rotate(_rotate(lr[:, h0:h1, :, :], axis=2, copy=False), axis=0, copy=False)
        x_rot2 = np.expand_dims(np.squeeze(x_rot2, -1), 1)        # (Hb, 1, W, D)
        a2 = _run(x_rot2)
        a2 = np.expand_dims(np.squeeze(a2, 1), -1)               # (Hb, W, D, 1)
        u2 = _rotate(_rotate(a2, -1, axis=0, copy=False), -1, axis=2, copy=False)
        isoim2[:, h0:h1, :, :] = u2

    sr = np.sqrt(np.maximum(isoim1, 0) * np.maximum(isoim2, 0))
    sr = np.squeeze(sr, -1)  # (D, H, W)
    return _percentile_denormalize(sr, mi, ma)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HeadSpec:
    op_name: str
    checkpoint: str          # path relative to --ckpt-dir
    in_chans: int
    upscale: int
    ndim: int                # expected spatial ndim of the input (2 or 3)
    runner: Callable[..., np.ndarray]
    description: str         # user-facing; surfaced via ProcessImage.GetOpNames
    labels: tuple            # organizational tags (task / dataset / structure)


# Descriptions below are written for the end user (a microscopist choosing an op),
# not the developer: they state the imaged structure/sample, the expected input,
# what the restoration does, and the output. They are surfaced verbatim through
# ProcessImage.GetOpNames alongside `labels` and an input-shape hint.
HEADS: dict[str, HeadSpec] = {
    spec.op_name: spec
    for spec in [
        HeadSpec(
            "sr_factin", "SwinIRF-actin/model_best181.pt", 1, 2, 2, _predict_sr,
            "Super-resolution of the actin cytoskeleton (F-actin). Give it a single "
            "2D wide-field fluorescence image of F-actin; it returns an image at 2x "
            "the input width and height, recovering fine filament detail toward "
            "SIM-like resolution. Trained on the BioSR F-actin data.",
            ("super-resolution", "BioSR", "F-actin"),
        ),
        HeadSpec(
            "sr_ccps", "SwinIRCCPs/model_best.pt", 1, 2, 2, _predict_sr,
            "Super-resolution of clathrin-coated pits (CCPs). Give it a single 2D "
            "wide-field fluorescence image of CCPs; it returns a 2x-larger image with "
            "individual pits more clearly resolved. Trained on the BioSR CCPs data.",
            ("super-resolution", "BioSR", "CCPs"),
        ),
        HeadSpec(
            "sr_er", "SwinIRER/model_best147.pt", 1, 2, 2, _predict_sr,
            "Super-resolution of the endoplasmic reticulum (ER). Give it a single 2D "
            "wide-field fluorescence image of the ER; it returns a 2x-larger image "
            "with sharper tubular network structure. Trained on the BioSR ER data.",
            ("super-resolution", "BioSR", "ER"),
        ),
        HeadSpec(
            "sr_microtubules", "SwinIRMicrotubules/model_best.pt", 1, 2, 2, _predict_sr,
            "Super-resolution of microtubules. Give it a single 2D wide-field "
            "fluorescence image of microtubules; it returns a 2x-larger image that "
            "separates closely spaced filaments. Trained on the BioSR microtubules data.",
            ("super-resolution", "BioSR", "microtubules"),
        ),
        HeadSpec(
            "denoise_planaria", "SwinIRDenoising_Planaria/model_best.pt", 1, 1, 3,
            _predict_denoise,
            "Denoising of 3D fluorescence stacks of whole Planaria (flatworm). Give it "
            "a low-SNR Z-stack (Z, Y, X); it returns a denoised stack of the same shape, "
            "each z-slice restored independently. Trained on the CARE Planaria data.",
            ("denoising", "CARE", "Planaria"),
        ),
        HeadSpec(
            "denoise_tribolium", "SwinIRmto1Denoising_Tribolium/model_best.pt", 5, 1, 3,
            _predict_denoise,
            "Denoising of 3D fluorescence stacks of Tribolium (flour-beetle) embryos. "
            "Give it a low-SNR Z-stack (Z, Y, X); it returns a denoised stack of the same "
            "shape, restoring each slice from a 5-slice neighbourhood for axial "
            "consistency. Trained on the CARE Tribolium data.",
            ("denoising", "CARE", "Tribolium"),
        ),
        HeadSpec(
            "isotropic_liver", "SwinIRIsotropic_Liver/model_best465.pt", 1, 1, 3,
            _predict_isotropic,
            "Isotropic reconstruction of 3D liver stacks. Give it an anisotropic Z-stack "
            "(Z, Y, X) whose axial sampling is coarser than its lateral sampling; it "
            "restores the under-sampled axial direction so resolution is more uniform in "
            "all three axes, returning a stack of the same shape. Trained on the CARE "
            "liver data.",
            ("isotropic-reconstruction", "CARE", "liver"),
        ),
    ]
}


def input_shape_hint(spec: HeadSpec) -> tuple[list[str], list[str]]:
    """(expected_singletons, required_multivalue) describing the op's input.

    2D ops want a single plane (Z singleton); 3D ops require a Z-stack.
    """
    if spec.ndim == 2:
        return ["Z"], []
    return [], ["Z"]


def load_head(spec: HeadSpec, ckpt_dir: str, device: torch.device):
    """Instantiate the SwinIR backbone for ``spec`` and load its checkpoint."""
    path = os.path.join(ckpt_dir, spec.checkpoint)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint for op '{spec.op_name}' not found at {path}. "
            f"Set --ckpt-dir or bake the checkpoint into the image."
        )
    logger.info("Loading head '%s' from %s", spec.op_name, path)
    model = SwinIR(upscale=spec.upscale, in_chans=spec.in_chans)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and "model_state" not in state:
        # Some checkpoints wrap the weights; bare state_dicts are the common case.
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def predict(spec: HeadSpec, image: np.ndarray, model, device: torch.device) -> np.ndarray:
    """Run the head's task-specific inference on a decoded (eager) image."""
    return spec.runner(image, model, device, spec)
