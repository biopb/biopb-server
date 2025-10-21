import logging
import random
from concurrent import futures
from pathlib import Path
from typing import Optional

import biopb.image as proto
import cv2
import grpc
import jax
import numpy as np
import typer
from flax import nnx
from biopb.image.utils import roi_to_mask
from skimage.measure import regionprops
from ctxseg.modeling.diffusion import edm_precond
from ctxseg.segmentation.flow import flow_to_mask
from ctxseg.segmentation.utils import pad_channel, clean_up_mask, remove_small_instances
from common import decode_image, encode_image, TokenValidationInterceptor, BiopbServicerBase

app = typer.Typer(pretty_exceptions_enable=False)

logger = logging.getLogger(__name__)

jnp = jax.numpy

rngs = nnx.Rngs(random.randint(0,10000))

_MAX_MSG_SIZE=1024*1024*128

def _to_patches(image, GS=512):
    overlap = GS // 8
    H, W, _ = image.shape
    
    stack = []
    for yc in range(0, H, GS-overlap):
        for xc in range(0, W, GS-overlap):
            patch = image[yc:yc+GS, xc:xc+GS]
            ph, pw, _ = patch.shape
            if ph > overlap and pw > overlap:
                patch = np.pad(patch, [[0, GS-ph],[0,GS-pw],[0,0]])
                stack.append(patch)
    return np.stack(stack)


@nnx.jit
def _model_fn(model, x_t, sigma, conditional=None):
    model_def, weights = nnx.split(model)

    def _f(x):
        _model = nnx.merge(model_def, weights)
        return edm_precond(_model.predict, sigma_data=1.0)(x, sigma)

    denoised, vjp_f = jax.vjp(_f, x_t)

    if conditional is not None:
        inpaint, mask = conditional
        tangents = jnp.where(mask, inpaint - denoised, 0)
        guidance = vjp_f(tangents)[0]
        denoised = denoised + sigma * guidance

    return denoised


def _predict(model, latent, sigma, *, GS=512, **kwargs):
    overlap = GS // 8
    H, W, _ = latent.shape

    latent = _to_patches(latent, GS=GS)
    stack = np.asarray(_model_fn(model, latent, sigma, **kwargs))

    cnts = np.zeros([H, W, 1])
    flow = np.zeros([H, W, 2])
    k = 0
    for yc in range(0, H, GS-overlap):
        for xc in range(0, W, GS-overlap):
            patch = flow[yc:yc+GS, xc:xc+GS]
            ph, pw, _ = patch.shape
            if ph > overlap and pw > overlap:
                cnts[yc:yc+GS, xc:xc+GS] += 1
                flow[yc:yc+ph, xc:xc+pw] += stack[k, :ph, :pw]
                k += 1

    flow = flow / cnts

    return flow


def _draw_sample(model, image, conditional=None, *, steps=4):
    sigma = 80
    H, W = image.shape[:2]

    if conditional is not None:
        inpaint, inpaint_mask = conditional
        conditional = (
            _to_patches(inpaint),
            _to_patches(inpaint_mask[:, :, None]),
        )

    latent = jax.random.normal(rngs.default(), (H, W, 2)) * sigma

    e_x = _predict(model, latent, sigma, conditional=conditional)

    for k in range(steps - 1):
        sigma_next = sigma / 2
        r = sigma_next / sigma
        latent = latent * r + (1-r) * e_x
        e_x = _predict(model, latent, sigma)
        sigma = sigma_next
    
    if conditional is not None:
        e_x = jnp.where(inpaint_mask[:, :, None], inpaint, e_x)

    return e_x


def _to_detections(label, flow):
    score = (flow ** 2).sum(axis=-1)
    response = proto.DetectionResponse()
    for rp in regionprops(label, score):
        mask = rp.image.astype("uint8")
        c, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = np.array(c[0], dtype=float).squeeze(1)
        c = c + np.array([rp.bbox[1] , rp.bbox[0]])
        c = c - 0.5

        scored_roi = proto.ScoredROI(
            score = rp.mean_intensity,
            roi = proto.ROI(
                polygon = proto.Polygon(points = [proto.Point(x=p[0], y=p[1]) for p in c]),
            )
        )

        response.detections.append(scored_roi)
    
    logger.debug(f"Found {len(response.detections)} detections")

    return response


def _format_image(image):
    image = image - image.min()
    image = image / (image.max() + 1e-3)
    image = pad_channel(image)
    return image


def _segment_image(model, image):
    model.set_image(_to_patches(image))

    flow = _draw_sample(model, image, steps=1)

    label = flow_to_mask(flow)
    label = clean_up_mask(label)

    return np.asarray(label), np.asarray(flow)


class CtxSegServicer(BiopbServicerBase):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def _process_data(self, pixels):
            image = decode_image(pixels) #[ZYXC]

            if image.shape[0] > 1:
                raise ValueError(f"Model does not support 3D input, got shape {image.shape}")

            logger.info(f"received image {image.shape}")

            image = _format_image(image[0])

            return image


    def RunDetection(self, request, context):
        with self._server_context(context):
            image = self._process_data(request.image_data.pixels)

            label, flow = _segment_image(self.model, image)

            response = _to_detections(label, flow)

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response


    def Run(self, request, context):
        with self._server_context(context):
            image = self._process_data(request.image_data.pixels)

            label, flow = _segment_image(self.model, image)

            response = proto.ProcessResponse(
                image_data = proto.ImageData(pixels = encode_image(label.astype('uint16'))),
            )

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response


    def RunStream(self, request_iterator, context):
        _ctx = None
        for request in request_iterator:
            with self._server_context(context):
                conditional = None

                if _ctx is None:
                    image = self._process_data(request.image_data.pixels)
                    self.model.set_image(_to_patches(image))
                    _ctx = self.model.x_skips

                elif len(request.image_data.image_annotation.rois) > 0:
                    template = np.zeros(image.shape[:2], dtype='uint8')
                    mask = [roi_to_mask(roi, template) for roi in request.image_data.image_annotation.rois]
                    logger.info(f'received mask {mask.shape}')
                    conditional = (flow, np.any(mask, axis=0))

                self.model.x_skips = _ctx

                flow = _draw_sample(self.model, image, conditional=conditional)

                label = flow_to_mask(flow)

                label = np.array(clean_up_mask(label), dtype='uint16')

                response = proto.ProcessResponse(
                    image_data = proto.ImageData(pixels = encode_image(label)),
                )

                yield response


def load_model(model_file):
    import pickle
    from ctxseg.modeling.diffusion import SegP

    params = pickle.load(open(model_file, "rb"))
    model = SegP(patch_size=2, rngs=rngs)
    nnx.update(model, params)

    logger.info(f'Loaded model {model_file}')

    return model


@app.command()
def main(
    modelpath: Path,
    port: int = 50051,
    workers: int = 10,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: Optional[bool] = None,
    debug: bool = False,
    compression: bool = True,
):
    print ("server starting ...")

    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    model = load_model(modelpath)

    if token is None:
        token = not local
    if token:
        import secrets

        token_str = secrets.token_urlsafe(64)

        print()
        print("COPY THE TOKEN BELOW FOR ACCESS.")
        print("=======================================================================")
        print(f"{token_str}")
        print("=======================================================================")
        print()
    else:
        token_str = None

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers),
        compression=grpc.Compression.Gzip if compression else grpc.Compression.NoCompression,
        interceptors=(TokenValidationInterceptor(token_str),),
        options=(("grpc.max_receive_message_length", _MAX_MSG_SIZE),),
    )

    servicer = CtxSegServicer(model)
    proto.add_ObjectDetectionServicer_to_server(servicer, server)
    proto.add_ProcessImageServicer_to_server(servicer, server)

    if local:
        server.add_secure_port(f"127.0.0.1:{port}", grpc.local_server_credentials())
    else:
        server.add_insecure_port(f"{ip}:{port}")

    logger.info(f"ctxseg_server: listening on port {port}")

    print ("server starting ... ready")

    server.start()

    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    app()
