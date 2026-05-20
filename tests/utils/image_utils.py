"""Test utilities for biopb-server."""

from pathlib import Path
import numpy as np
import imageio.v2 as imageio

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def load_test_image(name: str = "test_image.png") -> np.ndarray:
    """Load a test image from fixtures directory.

    Args:
        name: Image filename in fixtures directory

    Returns:
        Numpy array with image data
    """
    path = FIXTURES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Test image not found: {path}")
    return imageio.imread(str(path))


def generate_test_image(
    height: int = 512,
    width: int = 512,
    channels: int = 3,
    dtype: str = "uint8",
) -> np.ndarray:
    """Generate a synthetic test image.

    Args:
        height: Image height in pixels
        width: Image width in pixels
        channels: Number of channels (1 for grayscale, 3 for RGB)
        dtype: Data type ("uint8", "uint16", "float32")

    Returns:
        Numpy array with generated image data
    """
    if dtype == "uint8":
        return np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
    elif dtype == "uint16":
        return np.random.randint(0, 65535, (height, width, channels), dtype=np.uint16)
    elif dtype == "float32":
        return np.random.rand(height, width, channels).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def generate_multichannel_image(
    height: int = 512,
    width: int = 512,
    n_channels: int = 3,
) -> np.ndarray:
    """Generate a multi-channel fluorescence-style test image.

    Args:
        height: Image height
        width: Image width
        n_channels: Number of channels

    Returns:
        Numpy array (height, width, n_channels)
    """
    return np.random.randint(0, 255, (height, width, n_channels), dtype=np.uint8)


def generate_3d_stack(
    n_slices: int = 10,
    height: int = 256,
    width: int = 256,
    channels: int = 1,
) -> np.ndarray:
    """Generate a 3D image stack.

    Args:
        n_slices: Number of Z slices
        height: Image height
        width: Image width
        channels: Number of channels

    Returns:
        Numpy array (n_slices, height, width, channels)
    """
    return np.random.randint(0, 255, (n_slices, height, width, channels), dtype=np.uint8)