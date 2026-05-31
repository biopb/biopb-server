"""Unit tests for the non-overlapping chunk tiler (unifmr/tiling.py).

Container-free and GPU-free: a synthetic ``compute_chunk`` plays the role of the
model by returning the input region upscaled by nearest-neighbour repeat. This
isolates the tiling/upload orchestration and verifies the key invariants: the
assembled output equals the reference, and every output pixel is written exactly
once (no gaps, no double-writes / overlap).
"""

import numpy as np
import pytest

import tiling

pytestmark = pytest.mark.unit


def _upscale(a, up):
    """Nearest-neighbour upscale of the trailing two axes by ``up``."""
    return a.repeat(up, axis=-2).repeat(up, axis=-1)


def _run_case(img, tile_size, up, is_stack):
    if is_stack:
        zdim, height, width = img.shape
    else:
        height, width = img.shape
        zdim = None

    core = tiling.plane_core_shape((height, width), tile_size)

    if zdim is None:
        out = np.full((height * up, width * up), np.nan, dtype=float)
        cover = np.zeros((height * up, width * up), dtype=int)
    else:
        out = np.full((zdim, height * up, width * up), np.nan, dtype=float)
        cover = np.zeros((zdim, height * up, width * up), dtype=int)

    def compute_chunk(y0, y1, x0, x1):
        tile = img[y0:y1, x0:x1] if zdim is None else img[:, y0:y1, x0:x1]
        return _upscale(tile, up)

    def write_chunk(oy0, oy1, ox0, ox1, data):
        if zdim is None:
            out[oy0:oy1, ox0:ox1] = data
            cover[oy0:oy1, ox0:ox1] += 1
        else:
            out[:, oy0:oy1, ox0:ox1] = data
            cover[:, oy0:oy1, ox0:ox1] += 1

    n = tiling.tile_plane((height, width), core, up, compute_chunk, write_chunk)
    return out, cover, core, n


@pytest.mark.parametrize("up", [1, 2])
def test_2d_reconstruction_exact_once(up):
    rng = np.random.default_rng(0)
    img = rng.random((96, 120))
    out, cover, core, n = _run_case(img, tile_size=32, up=up, is_stack=False)

    # core must evenly divide the plane (tensor cache requires uniform chunks)
    assert 96 % core[0] == 0 and 120 % core[1] == 0
    assert n == (96 // core[0]) * (120 // core[1])
    assert np.array_equal(out, _upscale(img, up))   # correct assembly
    assert (cover == 1).all()                        # written exactly once


def test_3d_stack_reconstruction():
    rng = np.random.default_rng(1)
    img = rng.random((5, 64, 80))  # (Z, Y, X), upscale 1 for stacks
    out, cover, core, n = _run_case(img, tile_size=24, up=1, is_stack=True)

    assert 64 % core[0] == 0 and 80 % core[1] == 0
    assert np.array_equal(out, img)
    assert (cover == 1).all()


def test_uniform_core_picks_dividing_size():
    assert tiling.uniform_core(1024, 256) == 256
    assert tiling.uniform_core(96, 32) == 32
    assert tiling.uniform_core(50, 1024) == 50      # dim <= target -> whole dim
    assert tiling.uniform_core(97, 30) == 1         # prime -> only divisor < dim is 1
    c = tiling.uniform_core(100, 30)
    assert 100 % c == 0 and c == 25                 # closest divisor to 30
    # invariant: the result always evenly divides the dimension
    for dim in (256, 512, 384, 100, 97, 1000):
        assert dim % tiling.uniform_core(dim, 128) == 0
