import logging

import biopb.image as proto
import numpy as np
import typer

from common import decode_image, encode_image, BiopbServicerBase, setup_logging
from model import FinetunedSAM
from pipeline import SlidingWindowPipeline
from server import run_server

app = typer.Typer(pretty_exceptions_enable=False)

logger = logging.getLogger(__name__)

def process_input(request: proto.DetectionRequest | proto.ProcessRequest):
    logger.debug(f"Received message of size {request.ByteSize()}")

    pixels = request.image_data.pixels
    
    kwargs = dict()

    image = decode_image(pixels)
    physical_size = pixels.physical_size_x or 1

    if image.shape[0] == 1: # 2D
        image = image.squeeze(0)

    logger.info(f"Received imag of {image.shape}")
    
    return image, kwargs


def to_det_response(masks, image):
    import cv2
    from skimage.measure import regionprops

    response = proto.DetectionResponse()

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


class SamcellServicer(BiopbServicerBase):

    def __init__(self, model_path):
        super().__init__()
        model = FinetunedSAM('facebook/sam-vit-base', finetune_vision=False, finetune_prompt=True, finetune_decoder=True)
        model.load_weights(model_path)
        self.pipeline = SlidingWindowPipeline(model, 'cuda', crop_size=256)

    def _predict(self, image):
        if image.ndim == 4:
            raise ValueError("3D input is not supported.")
        
        image = image.mean(axis=-1)

        image = (image - image.min()) / (image.max() - image.min())

        image = (image * 255).astype('uint8')

        return self.pipeline.run(image)

    def RunDetection(self, request, context):
        with self._server_context(context):
            image, kwargs = process_input(request)
            
            mask = self._predict(image)

            logger.info(f"Reply mask image {mask.shape} with {mask.max()} labels.")

            response = to_det_response(mask, image)

            response = to_det_response(mask, image)

            return response


    def Run(self, request, context):
        with self._server_context(context):
            image, _ = process_input(request)

            mask = self._predict(image)
                
            response = proto.ProcessResponse(
                image_data = proto.ImageData(pixels = encode_image(mask)),
            )

            logger.debug(f"Reply with message of size {response.ByteSize()}")

            return response


@app.command()
def main(
    model_path: str,
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

    run_server(
        SamcellServicer(model_path=model_path),
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
