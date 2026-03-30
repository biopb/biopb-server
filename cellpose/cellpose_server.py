import logging

import biopb.image as proto
import numpy as np
import typer

from cellpose import models
from common import decode_image, encode_image, BiopbServicerBase, setup_logging, parse_kwargs, validate_kwargs
from server import run_server

app = typer.Typer(pretty_exceptions_enable=False)

logger = logging.getLogger(__name__)

_TARGET_CELL_SIZE=30

# Default kwargs for cellpose model
_DEFAULT_KWARGS = {
    "channels": [0, 0],
    "diameter": 30.0,
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "normalize": True,
    "invert": False,
    "min_size": 15,
}

# Validation schema for cellpose kwargs
_CELLPOSE_kwargs_SCHEMA = {
    "channels": {
        "type": "array",
        "item_type": "int",
        "min_length": 2,
        "max_length": 2,
        "description": "Channel specification [cytoplasm, nucleus]",
    },
    "diameter": {
        "type": "number",
        "minimum": 0,
        "description": "Cell diameter in pixels",
    },
    "flow_threshold": {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "description": "Flow error threshold",
    },
    "cellprob_threshold": {
        "type": "number",
        "minimum": -6.0,
        "maximum": 6.0,
        "description": "Cell probability threshold",
    },
    "normalize": {
        "type": "bool",
        "description": "Normalize image intensities",
    },
    "invert": {
        "type": "bool",
        "description": "Invert image pixel intensity",
    },
    "min_size": {
        "type": "int",
        "minimum": 0,
        "description": "Minimum cell size",
    },
}


def process_input(request: proto.DetectionRequest):
    pixels = request.image_data.pixels
    settings = request.detection_settings

    image = decode_image(pixels)

    physical_size = pixels.physical_size_x or 1

    # Start with default kwargs
    kwargs = _DEFAULT_KWARGS.copy()

    # Override diameter from detection_settings if provided
    if settings.HasField("cell_diameter_hint"):
        kwargs["diameter"] = settings.cell_diameter_hint / physical_size
    elif settings.scaling_hint:
        kwargs["diameter"] = _TARGET_CELL_SIZE / (settings.scaling_hint or 1.0)

    # Auto-detect channels if not specified in kwargs
    if image.shape[-1] > 1:
        kwargs["channels"] = [1, 2]
    else:
        kwargs["channels"] = [0, 0]

    # Merge with kwargs from request (if provided)
    kwargs = parse_kwargs(request, kwargs)

    # Validate kwargs
    errors = validate_kwargs(kwargs, _CELLPOSE_kwargs_SCHEMA)
    if errors:
        raise ValueError("Invalid kwargs: " + "; ".join(errors))

    if image.shape[0] == 1: # 2D
        image = image.squeeze(0)

    return image, kwargs


def process_result(preds, image):
    import cv2
    from skimage.measure import regionprops

    response = proto.DetectionResponse()

    try:
        masks, flows, styles, _ = preds
    except:
        masks, flows, styles = preds

    for rp in regionprops(masks):
        mask = rp.image.astype("uint8")
        c, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = np.array(c[0], dtype=float).squeeze(1)
        c = c + np.array([rp.bbox[1] , rp.bbox[0]])
        c = c - 0.5

        scored_roi = proto.ScoredROI(
            score = 1.0,
            roi = proto.ROI(
                polygon = proto.Polygon(points = [proto.Point(x=p[0], y=p[1]) for p in c]),
            )
        )

        response.detections.append(scored_roi)
    
    logger.debug(f"Found {len(response.detections)} detections")

    return response


class CellposeServicer(BiopbServicerBase):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def RunDetection(self, request, context):
        with self._server_context(context):
            logger.info(f"Received message of size {request.ByteSize()}")

            image, kwargs = process_input(request)

            if image.ndim == 4:
                raise ValueError("3D input not supported. Use the 'ProcessImage' service instead.")

            logger.info(f"received image {image.shape}")

            preds = self.model.eval(image,  **kwargs,)

            response = process_result(preds, image)

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response


    def Run(self, request, context):
        with self._server_context(context):
            logger.info(f"Received message of size {request.ByteSize()}")

            pixels = request.image_data.pixels
            image = decode_image(pixels)

            # Start with default kwargs
            kwargs = _DEFAULT_KWARGS.copy()

            if image.shape[0] == 1: # 2D
                image = image.squeeze(0)
            else:
                kwargs["do_3D"] = True

            # Merge with kwargs from request (if provided)
            kwargs = parse_kwargs(request, kwargs)

            # Validate kwargs
            errors = validate_kwargs(kwargs, _CELLPOSE_kwargs_SCHEMA)
            if errors:
                raise ValueError("Invalid kwargs: " + "; ".join(errors))

            logger.info(f"Decoded image {image.shape}")

            mask = self.model.eval(image, **kwargs)[0]

            response = proto.ProcessResponse(
                image_data = proto.ImageData(pixels = encode_image(mask)),
            )

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response

    def GetOpNames(self, request, context):
        """Return the available operations and their parameter schemas."""
        from google.protobuf.struct_pb2 import Struct

        default_kwargs = Struct()
        default_kwargs.update(_DEFAULT_KWARGS)

        schema = proto.OpSchema(
            default_kwargs=default_kwargs,
            description="Cellpose Cyto3 cell segmentation model",
        )

        op_names = proto.OpNames(names=["cellpose"])
        op_names.op_schemas.get_or_create("cellpose").CopyFrom(schema)

        return op_names


@app.command()
def main(
    modeltype: str = "cyto3",
    port: int = 50051,
    workers: int = 10,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: bool | None = None,
    debug: bool = False,
    json_logging: bool = False,
    compression: bool = True,
    gpu: bool = True,
):
    setup_logging(debug=debug, json_format=json_logging)

    model = models.Cellpose(model_type=modeltype, gpu=gpu)

    run_server(
        CellposeServicer(model),
        port=port,
        workers=workers,
        ip=ip,
        local=local,
        token=token,
        debug=debug,
        compression=compression,
    )


if __name__ == "__main__":
    app()
