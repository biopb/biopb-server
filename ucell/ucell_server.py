"""UCell gRPC server using biopb-base utilities."""

import logging
from pathlib import Path

import biopb.image as proto
import numpy as np
import torch
import typer
from ml_collections import ConfigDict

from ucell.dynamics import compute_masks, remove_bad_flow_masks
from ucell.frm import FRMWrapper
from ucell.utils import pad_channel, patcherize

from common import (
    BiopbServicerBase,
    decode_image,
    encode_image,
    parse_kwargs,
    setup_logging,
    validate_kwargs,
)

from server import run_server

app = typer.Typer(pretty_exceptions_enable=False)

logger = logging.getLogger(__name__)

# Default kwargs for ucell model
_DEFAULT_KWARGS = {
    "task_id": 0,
    "cellprob_threshold": -0.2,
    "min_area": 5,
}

# Validation schema for ucell kwargs
_KWARGS_SCHEMA = {
    "task_id": {
        "type": "int",
        "minimum": 0,
        "description": "Task ID for multi-task models",
    },
    "cellprob_threshold": {
        "type": "number",
        "minimum": -6.0,
        "maximum": 6.0,
        "description": "Cell probability logit threshold",
    },
    "min_area": {
        "type": "int",
        "minimum": 0,
        "description": "Minimum cell area",
    },
}

_NITER = 500  # Integration steps for mask computation

def get_config():
    """Return default ucell model configuration."""
    config = ConfigDict()
    config.name = "ucell"
    config.seed = 42
    config.image_size = 256
    config.task_id = 0

    config.model = ConfigDict()
    config.model.patch_size = 8
    config.model.forward_dtype = "bfloat16"
    config.model.pos_emb = "rope"
    config.model.hidden_size = 1024
    config.model.num_z_tokens = 64
    config.model.num_task_emb_tokens = 64
    config.model.depth = 2
    config.model.num_heads = config.model.hidden_size // 64
    config.model.num_tasks = 5
    config.model.seq_len = (config.image_size // config.model.patch_size) ** 2
    config.model.H_cycles = 1
    config.model.L_cycles = 21
    config.halt_max_steps = 1

    # LoRA configuration (disabled by default)
    config.lora = ConfigDict()
    config.lora.rank = 0
    config.lora.alpha = 1.0
    config.lora.dropout = 0.0
    config.lora.target_modules = ["SwiGLU"]

    return config


def format_image(img: np.ndarray) -> np.ndarray:
    """Normalize and pad image channels for ucell model."""
    img = img / (img.max() + 1e-5)
    img = pad_channel(img)
    return img


def process_input(request: proto.DetectionRequest):
    """Process input request and return image and kwargs."""
    pixels = request.image_data.pixels
    image = decode_image(pixels)

    # Handle 2D images
    if image.shape[0] == 1:
        image = image.squeeze(0)

    # Normalize and pad channels
    image = format_image(image)

    # Parse kwargs
    kwargs = parse_kwargs(request, _DEFAULT_KWARGS)

    # Validate kwargs
    errors = validate_kwargs(kwargs, _KWARGS_SCHEMA)
    if errors:
        raise ValueError("Invalid kwargs: " + "; ".join(errors))

    return image, kwargs


def process_result(mask: np.ndarray, image: np.ndarray) -> proto.DetectionResponse:
    """Convert mask to DetectionResponse with polygon ROIs."""
    import cv2
    from skimage.measure import regionprops

    response = proto.DetectionResponse()

    for rp in regionprops(mask):
        mask_region = rp.image.astype("uint8")
        contours, _ = cv2.findContours(mask_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        contour = np.array(contours[0], dtype=float).squeeze(1)
        contour = contour + np.array([rp.bbox[1], rp.bbox[0]])
        contour = contour - 0.5

        scored_roi = proto.ScoredROI(
            score=1.0,
            roi=proto.ROI(
                polygon=proto.Polygon(
                    points=[proto.Point(x=p[0], y=p[1]) for p in contour]
                ),
            ),
        )

        response.detections.append(scored_roi)

    logger.debug(f"Found {len(response.detections)} detections")

    return response


def compute_instance_masks(flow: np.ndarray, cell_prob: np.ndarray, kwargs: dict, device: torch.device) -> np.ndarray:
    """Compute instance masks from flow and cell probability."""
    mask = compute_masks(
        flow * 4.0,
        cell_prob,
        niter=_NITER,
        cellprob_threshold=kwargs["cellprob_threshold"],
        flow_threshold=0,
        min_size=kwargs["min_area"],
        max_size_fraction=0.4,
        device=device,
    )

    return mask


class UCellServicer(BiopbServicerBase):
    """UCell servicer implementing ObjectDetection and ProcessImage services."""

    def __init__(self, model, config, device: torch.device):
        super().__init__(use_lock=False)
        self.model = model
        self.config = config
        self.device = device

    def RunDetection(self, request, context):
        with self._server_context(context):
            logger.info(f"Received message of size {request.ByteSize()}")

            image, kwargs = process_input(request)

            logger.info(f"Received image {image.shape}")

            # Run inference with patcherize for large images
            predict_fn = patcherize(self.model.predict, GS=self.config.image_size)
            with torch.device(self.device):
                output = predict_fn(image, kwargs["task_id"])

            # output shape: (H, W, 3) - flow[:2], cell_prob[2]
            flow = np.moveaxis(output[:, :, :2], -1, 0)  # (2, H, W)
            cell_prob = output[:, :, 2]

            mask = compute_instance_masks(flow, cell_prob, kwargs, self.device)

            response = process_result(mask, image)

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response

    def Run(self, request, context):
        with self._server_context(context):
            logger.info(f"Received message of size {request.ByteSize()}")

            image = decode_image(request.image_data.pixels)

            if image.shape[0] == 1:  # 2D
                image = image.squeeze(0)

            image = format_image(image)

            kwargs = parse_kwargs(request, _DEFAULT_KWARGS)
            errors = validate_kwargs(kwargs, _KWARGS_SCHEMA)
            if errors:
                raise ValueError("Invalid kwargs: " + "; ".join(errors))

            logger.info(f"Received image {image.shape}")

            predict_fn = patcherize(self.model.predict, GS=self.config.image_size)
            with torch.device(self.device):
                output = predict_fn(image, kwargs["task_id"])

            flow = np.moveaxis(output[:, :, :2], -1, 0)
            cell_prob = output[:, :, 2]

            mask = compute_instance_masks(flow, cell_prob, kwargs, self.device)

            logger.info(f"Detected {mask.max()} cells")

            response = proto.ProcessResponse(
                image_data=proto.ImageData(pixels=encode_image(mask.astype(np.uint16))),
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
            description="UCell cell segmentation model (FRM-based)",
        )

        op_names = proto.OpNames(names=["ucell"])
        op_names.op_schemas.get_or_create("ucell").CopyFrom(schema)

        return op_names


def load_model(modelpath: Path, config, device: torch.device):
    """Load and initialize the UCell model."""
    model = FRMWrapper(config).eval()

    default_lora_scaling = None
    if hasattr(config, "lora") and config.lora.rank > 0:
        default_lora_scaling = config.lora.alpha / config.lora.rank

    model.load_checkpoint(
        str(modelpath),
        default_lora_scaling=default_lora_scaling,
    )

    model = model.inner.to(device)

    if device.type == "cuda":
        model = torch.compile(model)

    logger.info(f"Loaded model from {modelpath}")

    return model


@app.command()
def main(
    modelpath: Path = typer.Option(..., help="Path to the model checkpoint"),
    port: int = 50051,
    workers: int = 10,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: bool | None = None,
    debug: bool = False,
    compression: bool = True,
    gpu: bool = True,
    json_logging: bool = False,
    small_model: bool = False,
):
    setup_logging(debug=debug, json_format=json_logging)

    config = get_config()
    if small_model: config.model.hidden_size = 768

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model = load_model(modelpath, config, device)

    logger.info(f"Default backend: {device}")

    if device.type == "cpu":
        logger.warning("WARNING: No GPU configuration. This might be very slow...")

    run_server(
        UCellServicer(model, config, device),
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
