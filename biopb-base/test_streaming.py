"""Test script for biopb gRPC streaming services."""

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
    iterations: int = 4,
):
    setup_logging(debug=debug)

    SERVER = f"{ip}:{port}"
    METADATA = ((_AUTH_HEADER_KEY, "Bearer " + token.strip()),)
    logger.info(f"Testing streaming server at {SERVER}")

    # Test health check first
    if health:
        logger.info("Testing health check...")
        with grpc.insecure_channel(SERVER) as channel:
            if not test_health(channel, METADATA):
                logger.error("Health check failed, server may not be ready")
                return

    test_image = imageio.imread(image_path)
    logger.info(f"Loaded image {image_path} with shape {test_image.shape}")

    def _stream_messages(image, n: int = 4):
        """Generate streaming request messages."""
        yield proto.ProcessRequest(
            image_data=proto.ImageData(pixels=encode_image(image)),
        )
        for _ in range(n - 1):
            yield proto.ProcessRequest()

    def _test_streaming(image, label: str = ""):
        logger.info(f"Testing streaming {label or 'default'}: shape={image.shape}")
        start_time = time.perf_counter()

        try:
            with grpc.insecure_channel(SERVER) as channel:
                stub = proto.ProcessImageStub(channel)
                response_count = 0
                for response in stub.RunStream(
                    _stream_messages(image, iterations),
                    metadata=METADATA,
                    timeout=60,
                ):
                    response_count += 1
                    result = decode_image(response.image_data.pixels)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    print(f"  Response {response_count}: {result.max()} labels in {elapsed_ms:.1f}ms")

            return True

        except grpc.RpcError as e:
            logger.error(f"Streaming call failed: {e}")
            return False

    # Test with different image sizes
    results = []
    results.append(_test_streaming(test_image, "original"))

    cropped = test_image[:373, :372]
    results.append(_test_streaming(cropped, "cropped"))

    padded = np.pad(test_image, [[0, 128], [0, 128]])
    results.append(_test_streaming(padded, "padded"))

    if all(results):
        print("\nAll streaming tests passed!")
    else:
        print("\nSome streaming tests failed!")
        raise SystemExit(1)


if __name__ == "__main__":
    app()