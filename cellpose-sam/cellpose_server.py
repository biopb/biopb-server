import logging

import biopb.image as proto
import numpy as np
import typer

from cellpose import models, io
from biopb_image_base import decode_image_data, encode_image, BiopbServicerBase, run_server, ensure_eager

app = typer.Typer(pretty_exceptions_enable=False)

logger = logging.getLogger(__name__)

_TARGET_CELL_SIZE=30


def process_input(request: proto.DetectionRequest | proto.ProcessRequest):
    logger.debug(f"Received message of size {request.ByteSize()}")

    image = decode_image_data(request.image_data)
    image = ensure_eager(image)

    pixels = request.image_data.pixels
    if isinstance(request, proto.DetectionRequest):
        settings = request.detection_settings

        physical_size = pixels.physical_size_x or 1

        if settings.HasField("cell_diameter_hint"):
            diameter = settings.cell_diameter_hint / physical_size
        else:
            diameter = _TARGET_CELL_SIZE / (settings.scaling_hint or 1.0)

        kwargs = dict(diameter = diameter)

        if settings.HasField("min_score"):
            kwargs.update(dict(cellprob_threshold=settings.min_score))

    else:
        kwargs = dict()

    physical_size = pixels.physical_size_x or 1

    if image.ndim == 2:  # 2D grayscale, add channel dimension
        image = image[..., np.newaxis]

    if image.shape[0] == 1: # leading singleton dimension
        image = image.squeeze(0)

    logger.debug(f"decoded image {image.shape}")

    logger.debug(f"kwargs: {kwargs}")

    return image, kwargs


def process_result(masks, image):
    import cv2
    from skimage.measure import regionprops

    response = proto.DetectionResponse()

    for rp in regionprops(masks):
        mask = rp.image.astype("uint8")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        # Use the largest contour
        contour = max(contours, key=len)

        # Need at least 3 points for a valid polygon
        if len(contour) < 3:
            continue

        # Extract points and shift to global coordinates
        points = contour.squeeze(1)  # Shape: (N, 2)
        points = points + np.array([rp.bbox[1], rp.bbox[0]])
        points = points - 0.5

        scored_roi = proto.ScoredROI(
            score=1.0,
            roi=proto.ROI(
                polygon=proto.Polygon(points=[proto.Point(x=p[0], y=p[1]) for p in points]),
            ),
        )

        response.detections.append(scored_roi)

    logger.debug(f"Found {len(response.detections)} detections")

    logger.debug(f"Reply with message of size {response.ByteSize()}")

    return response


class CellposeServicer(BiopbServicerBase):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def GetOpNames(self, request, context):
        """Return the available operations and their parameter schemas."""
        with self._server_context(context):
            return proto.OpNames(
                names=["cellpose"],
                op_schemas={
                    "cellpose": proto.OpSchema(
                        description="Cellpose cell segmentation model",
                    ),
                }
            )

    def RunDetection(self, request, context):
        with self._server_context(context):
            image, kwargs = process_input(request)

            if image.ndim == 4:
                raise ValueError("3D input not supported. Use the 'ProcessImage' service instead.")

            logger.info(f"call model with image {image.shape}")

            masks = self.model.eval(image, **kwargs)[0]
            
            logger.info(f"received masks {masks.shape} with {masks.max()} labels.")

            response = process_result(masks, image)

            return response


    def Run(self, request, context):
        with self._server_context(context):
            image, kwargs = process_input(request)

            if image.ndim == 3: # 2D
                mask = self.model.eval(image, **kwargs)[0]
            else:
                mask = self.model.eval(
                    image, 
                    channel_axis=-1,
                    z_axis=0,
                    do_3D=True, 
                    flow3D_smooth=1, 
                    **kwargs
                )[0]
                
            response = proto.ProcessResponse(
                image_data = encode_image(mask),
            )

            logger.debug(f"Reply with message of size {response.ByteSize()}")

            return response


@app.command()
def main(
    modeltype: str = "",  # ignored, kept for compatibility
    port: int = 50051,
    workers: int = 10,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: bool | None = None,
    debug: bool = False,
    compression: bool = True,
    gpu: bool = True,
):
    model = models.CellposeModel(gpu=gpu)
    io.logger_setup()

    run_server(
        CellposeServicer(model),
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
