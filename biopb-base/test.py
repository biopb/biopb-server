"""Test script for biopb gRPC services."""

import logging
import time
from pathlib import Path

import biopb.image as proto
import grpc
import imageio.v2 as imageio
import numpy as np
import typer
from grpc_health.v1 import health_pb2, health_pb2_grpc

from common import decode_image, encode_image, _AUTH_HEADER_KEY, setup_logging

app = typer.Typer(pretty_exceptions_enable=False)

logger = logging.getLogger(__name__)


def construct_request(image: np.ndarray) -> proto.DetectionRequest:
    return proto.DetectionRequest(
        image_data=proto.ImageData(pixels=encode_image(image)),
        detection_settings=proto.DetectionSettings(
            scaling_hint=1.0,
        ),
    )


def test_health(channel: grpc.Channel, metadata: tuple) -> bool:
    """Test the health check endpoint."""
    try:
        stub = health_pb2_grpc.HealthStub(channel)
        request = health_pb2.HealthCheckRequest()
        response = stub.Check(request, metadata=metadata, timeout=5)
        status = health_pb2.HealthCheckResponse.ServingStatus.Name(response.status)
        print(f"Health check: {status}")
        return response.status == health_pb2.HealthCheckResponse.SERVING
    except grpc.RpcError as e:
        logger.error(f"Health check failed: {e}")
        return False


@app.command()
def main(
    port: int = 50051,
    ip: str = "127.0.0.1",
    token: str = "",
    image_path: Path = Path(__file__).parent / "test_image.png",
    debug: bool = False,
    health: bool = True,
):
    setup_logging(debug=debug)

    SERVER = f"{ip}:{port}"
    METADATA = ((_AUTH_HEADER_KEY, "Bearer " + token.strip()),)
    logger.info(f"Testing server at {SERVER}")

    # Test health check first
    if health:
        logger.info("Testing health check...")
        with grpc.insecure_channel(SERVER) as channel:
            if not test_health(channel, METADATA):
                logger.error("Health check failed, server may not be ready")
                return

    test_image = imageio.imread(image_path)
    logger.info(f"Loaded image {image_path} with shape {test_image.shape}")

    def _test_with_image(image, label: str = ""):
        logger.info(f"Testing image {label or 'default'}: shape={image.shape}")
        start_time = time.perf_counter()

        # Test ObjectDetection service
        try:
            with grpc.insecure_channel(SERVER) as channel:
                stub = proto.ObjectDetectionStub(channel)
                response = stub.RunDetection(
                    proto.DetectionRequest(
                        image_data=proto.ImageData(pixels=encode_image(image)),
                        detection_settings=proto.DetectionSettings(),
                    ),
                    metadata=METADATA,
                    timeout=30,
                )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(
                f"  ObjectDetection: {len(response.detections)} detections "
                f"in {elapsed_ms:.1f}ms"
            )

        except grpc.RpcError as e:
            logger.error(f"ObjectDetection call failed: {e}")
            return False

        # Test ProcessImage service
        start_time = time.perf_counter()
        try:
            with grpc.insecure_channel(SERVER) as channel:
                stub = proto.ProcessImageStub(channel)
                response = stub.Run(
                    proto.ProcessRequest(
                        image_data=proto.ImageData(pixels=encode_image(image)),
                    ),
                    metadata=METADATA,
                    timeout=30,
                )

            result = decode_image(response.image_data.pixels)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"  ProcessImage: {result.max()} labels in {elapsed_ms:.1f}ms")
            return True

        except grpc.RpcError as e:
            logger.error(f"ProcessImage call failed: {e}")
            return False

    # Test with different image sizes
    results = []
    results.append(_test_with_image(test_image, "original"))

    cropped = test_image[:373, :372]
    results.append(_test_with_image(cropped, "cropped"))

    padded = np.pad(test_image, [[0, 128], [0, 128]])
    results.append(_test_with_image(padded, "padded"))

    if all(results):
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        raise SystemExit(1)


if __name__ == "__main__":
    app()