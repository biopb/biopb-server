"""Non-overlapping chunk tiling for lazy (larger-than-memory) pixel->pixel ops.

The lazy backend does not (yet) implement halo / overlapping read-write, so each
chunk is processed independently with **no overlap and no blending**: every
output pixel is written exactly once by exactly one chunk. Seams between chunks
are accepted for now; feathered blending can be added later if the backend gains
halo support.

The orchestration is expressed through callbacks (like ``ucell/stitch.py``) so it
can be unit-tested without a model or a tensor cache.
"""

from __future__ import annotations

from typing import Callable

# Shared chunk-sizing primitive (issue #1): formerly a byte-identical local copy,
# now the single source of truth in biopb_image_base.stitch.
from biopb_image_base.stitch import uniform_core


def plane_core_shape(plane_shape: tuple, tile_size: int) -> tuple:
    """Per-axis core size for a Y/X plane, each evenly dividing its dimension."""
    return tuple(uniform_core(int(plane_shape[i]), tile_size) for i in range(len(plane_shape)))


def tile_plane(
    plane_shape: tuple,
    core_shape: tuple,
    upscale: int,
    compute_chunk: Callable[[int, int, int, int], object],
    write_chunk: Callable[[int, int, int, int, object], None],
) -> int:
    """Process a Y/X plane in non-overlapping ``core_shape`` chunks.

    Args:
        plane_shape: ``(H, W)`` input spatial shape.
        core_shape:  ``(ch, cw)`` input chunk size (should divide ``plane_shape``).
        upscale:     output-to-input spatial scale (2 for SR, 1 otherwise). The
                     output region is the input region multiplied by this factor.
        compute_chunk: ``(y0, y1, x0, x1) -> output_array`` for that input region;
                     the returned array's trailing two axes must be
                     ``((y1-y0)*upscale, (x1-x0)*upscale)``.
        write_chunk: ``(oy0, oy1, ox0, ox1, output_array) -> None`` stores the
                     output region at the upscaled bounds.

    Returns:
        The number of chunks processed.
    """
    height, width = int(plane_shape[0]), int(plane_shape[1])
    ch, cw = int(core_shape[0]), int(core_shape[1])
    count = 0
    for y0 in range(0, height, ch):
        y1 = min(y0 + ch, height)
        for x0 in range(0, width, cw):
            x1 = min(x0 + cw, width)
            data = compute_chunk(y0, y1, x0, x1)
            write_chunk(y0 * upscale, y1 * upscale, x0 * upscale, x1 * upscale, data)
            count += 1
    return count
