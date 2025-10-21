import logging

import biopb.image as proto
import imageio.v2 as imageio
import grpc
import numpy as np
import typer

from pathlib import Path
from common import decode_image, encode_image, _AUTH_HEADER_KEY

app = typer.Typer(pretty_exceptions_enable=False)

logger = logging.getLogger(__name__)

def construct_request(image: np.ndarray) -> proto.DetectionRequest:
    return proto.DetectionRequest(
        image_data = proto.ImageData(pixles = encode_image(image)),
        detection_settings = proto.DetectionSettings(
          scaling_hint = 1.0,
        ),
    )


@app.command()
def main(
    port: int = 50051,
    ip: str = "127.0.0.1",
    token: str = "",
    image_path: Path = Path("test_image.png"),
    debug: bool = False,
):
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    SERVER = f"{ip}:{port}"
    METADATA = ((_AUTH_HEADER_KEY, "Bearer " + token.strip()),) 
    logger.info(f"Testing server at {SERVER} with token {token}")

    test_image = imageio.imread(image_path)
    logger.info(f"Loaded image {image_path} with shape {test_image.shape}")

    def _messages(image, n = 4):
        yield proto.ProcessRequest(
            image_data=proto.ImageData(pixels=encode_image(image)),
        )
        for _ in range(n - 1):
            yield proto.ImageData()

    def _test_with_image(image):
        try:
            with grpc.insecure_channel(SERVER) as channel:
                stub = proto.ProcessImageStub(channel)
                for response in stub.RunStream(_messages(image), metadata=METADATA):
                    result = decode_image(response.image_data.pixels)
                    print(f"ProcessImage call responsed with {result.max()} detections.")

        except grpc.RpcError as e:
            logger.error(f"ProcessImage call failed: {e}")
            return
    
    _test_with_image(test_image)

    cropped = test_image[:373, :372]
    logger.info(f"Testing image size {cropped.shape}")
    _test_with_image(cropped)

    padded = np.pad(test_image, [[0, 128], [0, 128]])
    logger.info(f"Testing image size {padded.shape}")
    _test_with_image(padded)

if __name__ == "__main__":
    logging.basicConfig()
    app()
