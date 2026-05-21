"""Lacss gRPC server using biopb_image_base utilities."""

import logging
from pathlib import Path
from typing import Iterable

import biopb.image as proto
import jax
import numpy as np
import typer
from biopb.image.utils import serialize_from_numpy

from biopb_image_base import decode_image_data, BiopbServicerBase, run_server, ensure_eager

app = typer.Typer(pretty_exceptions_enable=False)

logger = logging.getLogger(__name__)

_TARGET_CELL_SIZE = 32


def _get_physical_sizes(image_data: proto.ImageData) -> np.ndarray:
    """Get physical sizes from ImageData, handling both pixels and eager_data formats.

    Returns physical sizes for Z, Y, X as a numpy array.
    Defaults to 1.0 if not specified.
    """
    # Check deprecated pixels field first
    if image_data.HasField("pixels"):
        pixels = image_data.pixels
        physical_size = np.array(
            [
                pixels.physical_size_z or pixels.physical_size_x,  # one might set xy but not z
                pixels.physical_size_y or 1.0,
                pixels.physical_size_x or 1.0,
            ],
            dtype="float",
        )
    # Check image_annotation for new format
    elif image_data.HasField("image_annotation") and len(image_data.image_annotation.pixels_sizes) >= 2:
        sizes = image_data.image_annotation.pixels_sizes
        # pixels_sizes order should match dimension order, typically [Z, Y, X] or [Y, X]
        if len(sizes) >= 3:
            physical_size = np.array([sizes[0], sizes[1], sizes[2]], dtype="float")
        else:
            # Assume Y, X only
            physical_size = np.array([sizes[0], sizes[0], sizes[1]], dtype="float")
    else:
        physical_size = np.array([1.0, 1.0, 1.0], dtype="float")

    # Ensure no zeros
    physical_size[physical_size == 0] = 1.0

    return physical_size


def _process_input(request: proto.DetectionRequest, image=None):
    """Process input request and return image and kwargs for lacss predict."""
    settings = request.detection_settings

    if image is None:
        image = decode_image_data(request.image_data)
        image = ensure_eager(image)

    physical_size = _get_physical_sizes(request.image_data)

    # Get image dimensions - handle 2D (H, W) or 2D with channel (H, W, C)
    # or 3D (Z, H, W) / (Z, H, W, C)
    if image.ndim == 2:  # Pure 2D grayscale (H, W)
        img_shape_3d = (1, image.shape[0], image.shape[1])
        phys_size_3d = (physical_size[0] if len(physical_size) > 2 else 1.0, physical_size[1], physical_size[2] if len(physical_size) > 2 else physical_size[1])
    elif image.ndim == 3:  # Could be 2D+channel or 3D
        # Assume last dim is channel if it's small (1-4), else it's 3D without channel
        if image.shape[-1] <= 4:
            img_shape_3d = (1, image.shape[0], image.shape[1])
            phys_size_3d = (physical_size[0] if len(physical_size) > 2 else 1.0, physical_size[1], physical_size[2] if len(physical_size) > 2 else physical_size[1])
        else:
            img_shape_3d = image.shape[:3]
            phys_size_3d = physical_size[:3]
    else:  # 4D or more - assume (Z, Y, X, C)
        img_shape_3d = image.shape[:3]
        phys_size_3d = physical_size[:3]

    phys_size_3d = np.array(phys_size_3d)

    if settings.HasField("cell_diameter_hint"):
        scaling = _TARGET_CELL_SIZE / settings.cell_diameter_hint * phys_size_3d

    else:
        if phys_size_3d[1] != phys_size_3d[2]:
            raise ValueError("Scaling hint provided, but pixel is not isometric")

        scaling = np.array([settings.scaling_hint or 1.0] * 3, dtype="float")
        scaling[0] *= phys_size_3d[0] / phys_size_3d[1]

    logger.info(f"Requested rescaling factor is {scaling}")

    shape_hint = tuple(np.round(scaling * img_shape_3d).astype(int))

    # Handle 2D images
    if image.ndim == 2:
        # Pure 2D - squeeze Z dimension from shape_hint
        shape_hint = shape_hint[1:]
    elif image.ndim == 3 and image.shape[-1] <= 4:
        # 2D with channel - squeeze Z dimension from shape_hint
        shape_hint = shape_hint[1:]
        # But keep channel for reshape_to
        if image.shape[-1] > 1:
            shape_hint = list(shape_hint) + [image.shape[-1]]

    kwargs = dict(
        reshape_to=shape_hint,
        score_threshold=settings.min_score or 0.4,
        min_area=settings.min_cell_area,
        nms_iou=settings.nms_iou or 0.4,
        segmentation_threshold=settings.segmentation_threshold or 0.5,
    )

    return image, kwargs


def _process_result(preds, image) -> proto.DetectionResponse:
    """Convert lacss predictions to DetectionResponse."""
    response = proto.DetectionResponse()

    is_2d = image.ndim == 2 or (image.ndim == 3 and image.shape[-1] <= 4)

    if is_2d:  # 2D returns polygon
        for contour, score in zip(preds["pred_contours"], preds["pred_scores"]):
            if len(contour) == 0:
                continue

            scored_roi = proto.ScoredROI(
                score=score,
                roi=proto.ROI(
                    polygon=proto.Polygon(
                        points=[proto.Point(x=p[0], y=p[1]) for p in contour]
                    ),
                ),
            )

            response.detections.append(scored_roi)

    else:  # 3D returns Mesh
        for mesh, score in zip(preds["pred_contours"], preds["pred_scores"]):
            scored_roi = proto.ScoredROI(
                score=score,
                roi=proto.ROI(
                    mesh=proto.Mesh(
                        verts=[
                            proto.Point(z=v[0], y=v[1], x=v[2]) for v in mesh["verts"]
                        ],
                        faces=[
                            proto.Mesh.Face(p1=p[0], p2=p[1], p3=p[2])
                            for p in mesh["faces"]
                        ],
                    ),
                ),
            )

            response.detections.append(scored_roi)

    return response


class LacssServicer(BiopbServicerBase):
    """Lacss servicer implementing ObjectDetection and ProcessImage services."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def RunDetection(self, request, context):
        with self._server_context(context):
            logger.info(f"Received message of size {request.ByteSize()}")

            image, kwargs = _process_input(request)

            logger.info(f"Received image {image.shape}")

            preds = self.model.predict(
                image,
                output_type="contour",
                **kwargs,
            )

            response = _process_result(preds, image)

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response

    def Run(self, request, context):
        with self._server_context(context):
            logger.info(f"Received message of size {request.ByteSize()}")

            image = decode_image_data(request.image_data)
            image = ensure_eager(image)

            if image.shape[0] == 1:  # 2D
                image = image.squeeze(0)

            logger.info(f"Received image {image.shape}")

            label = self.model.predict(image)["pred_label"]

            logger.info(f"Detected {label.max()} cells")

            response = proto.ProcessResponse(
                image_data=proto.ImageData(pixels=serialize_from_numpy(label)),
            )

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response

    def RunDetectionOnGrid(self, request_iterator: Iterable[proto.DetectionRequest], context):
        """Handle tiled/grid-based detection for large images."""
        with self._server_context(context):
            image, kwargs = _process_grid_input(request_iterator)

            if image is None:
                return proto.DetectionResponse()

            logger.info(f"Received full image of size {image.shape[:-1]}")

            preds = self.model.predict(
                image,
                output_type="contour",
                **kwargs,
            )

            response = _process_result(preds, image)

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response


def _process_grid_input(request_iterator: Iterable[proto.DetectionRequest]):
    """Process grid/tiled input requests."""
    _MAX_STREAM_MSG_SIZE = 1024 * 1024 * 128 * 16
    MAX_IMG_SIZE = 1024 * 1024 * 1024

    d, h, w, c = 0, 0, 0, 0
    images, grids = [], []
    request = None
    total_msg_size = 0

    for request in request_iterator:
        total_msg_size += request.ByteSize()
        if total_msg_size > _MAX_STREAM_MSG_SIZE:
            raise ValueError(f"Input message size {total_msg_size} exceeded limit.")

        image = decode_image_data(request.image_data)
        image = ensure_eager(image)
        pixels = request.image_data.pixels

        grids.append(
            [
                pixels.offset_z,
                pixels.offset_y,
                pixels.offset_x,
            ]
        )
        images.append(image)

        d = max(d, pixels.offset_z + image.shape[0])
        h = max(h, pixels.offset_y + image.shape[1])
        w = max(w, pixels.offset_x + image.shape[2])
        c = max(c, image.shape[3])

        if d * h * w > MAX_IMG_SIZE:
            raise ValueError(f"Input image is too large {(d, h, w)}")

    # Empty input iterator
    if request is None:
        return None, None

    assert c <= 3

    full_image = np.zeros([d, h, w, c], dtype=images[0].dtype)
    for image, grid in zip(images, grids):
        full_image[
            grid[0]: grid[0] + image.shape[0],
            grid[1]: grid[1] + image.shape[1],
            grid[2]: grid[2] + image.shape[2],
            : image.shape[3],
        ] = image

    return _process_input(request, full_image)


def get_predictor(modelpath: Path, f16: bool = False):
    """Load and initialize the Lacss predictor."""
    from lacss.deploy.predict import Predictor

    model = Predictor(modelpath, f16=f16)

    logger.info(f"Loaded model from {modelpath}")

    # Configure minimum score thresholds
    model.module.detector.min_score = 0.2
    if model.module.detector_3d:
        model.module.detector_3d.min_score = 0.2

    # Precompile the model (2D only to save GPU memory)
    logger.debug("Precompiling the model...")
    _ = model.predict(np.ones([256, 256, 3]), output_type="_raw")

    return model


@app.command()
def main(
    modelpath: Path = typer.Option("./lacss3-base.pkl", help="Path to the model file"),
    port: int = 50051,
    workers: int = 10,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: bool | None = None,
    debug: bool = False,
    compression: bool = True,
    f16: bool = False,
):
    model = get_predictor(modelpath, f16)

    logger.info(f"Default backend is {jax.default_backend()}")

    if jax.default_backend() == "cpu":
        logger.warning("WARNING: No GPU configuration. This might be very slow...")

    run_server(
        LacssServicer(model),
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