import logging
from pathlib import Path

import biopb.image as proto
import numpy as np
import typer

from flax import nnx

from common import decode_image, encode_image, BiopbServicerBase, setup_logging
from server import run_server

app = typer.Typer(pretty_exceptions_enable=False)

logger = logging.getLogger(__name__)


def process_input(request: proto.DetectionRequest):
    pixels = request.image_data.pixels
    settings = request.detection_settings

    image = decode_image(pixels).squeeze(0)

    return image


def process_result(masks):
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

    return response


class CtxSegServicer(BiopbServicerBase):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def predict(self, image, GS=512):
        from ctxseg.segmentation.utils import pad_channel
        from ctxseg.segmentation.flow import flow_to_mask
        from ctxseg.segmentation.utils import clean_up_mask, remove_small_instances


        @nnx.jit
        def _predict(model, image):
            return model.predict(image)

        image = (image - image.min()) / (image.max() - image.min())
        image = pad_channel(image)
        H, W, _ = image.shape

        flow = np.zeros([H, W, 3])
        cnts = np.zeros([H, W, 1])
        for yc in range(0, H, GS-64):
            for xc in range(0, W, GS-64):
                patch = image[yc:yc+GS, xc:xc+GS]
                ph, pw, _ = patch.shape
                if ph <= 64 or pw <= 64:
                    continue
                patch = np.pad(patch, [[0, GS-ph], [0, GS-pw], [0,0]])
                pred = _predict(self.model, patch[None])[0, :ph, :pw]
                flow[yc:yc+ph, xc:xc+pw] += pred
                cnts[yc:yc+ph, xc:xc+pw] += 1

        flow = flow / cnts
   
        mask = flow_to_mask(flow[..., :2])
        mask = np.where(flow[...,-1] >= 0, mask, 0)
        mask = clean_up_mask(remove_small_instances(mask, 100))

        return mask.astype('uint16')


    def RunDetection(self, request, context):
        with self._server_context(context):
            image = process_input(request)

            if image.ndim == 4:
                raise ValueError(f"Model does not support 3D input, got shape {image.shape}")

            logger.info(f"received image {image.shape}")

            preds = self.predict(image)

            response = process_result(preds)

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response


    def Run(self, request, context):
        with self._server_context(context):
            logger.info(f"Received message of size {request.ByteSize()}")

            pixels = request.image_data.pixels

            image = decode_image(pixels)

            logger.info(f"Decoded image {image.shape}")

            if image.shape[0] == 1: # 2D
                image = image.squeeze(0)
                mask = self.predict(image)

            else:
                raise ValueError(f"Model does not support 3D input, got shape {image.shape}")

            response = proto.ProcessResponse(
                image_data = proto.ImageData(pixels = encode_image(mask)),
            )

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response



def load_model(cp):
    import pickle

    with open(cp, "rb") as f:
        target = pickle.load(f)

    if isinstance(target, nnx.State):
        from ctxseg.modeling.ctxseg import CtxSegD
        rngs = nnx.Rngs(0, dropout=1234)
        model = CtxSegD(ps=2,rngs=rngs)
        nnx.update(model, target)
        target = model
    else:
        assert isinstance(target, nnx.Model), f"Unknown checkpoint format: {type(target)}"
    
    target.eval()

    return target

@app.command()
def main(
    modelpath: Path,
    port: int = 50051,
    workers: int = 10,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: bool | None = None,
    debug: bool = False,
    json_logging: bool = False,
    compression: bool = True,
):
    setup_logging(debug=debug, json_format=json_logging)

    model = load_model(modelpath)

    run_server(
        CtxSegServicer(model),
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
