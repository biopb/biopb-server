"""CLI test client for biopb.image services."""

import json
import time
from pathlib import Path

import typer
import grpc
import imageio.v2 as imageio
import numpy as np
from grpc_health.v1 import health_pb2, health_pb2_grpc

import biopb.image as proto
from biopb.image.utils import serialize_from_numpy_to_image_data, deserialize_image_data

app = typer.Typer(pretty_exceptions_enable=False)

# Large message options
_GRPC_OPTIONS = [
    ("grpc.max_receive_message_length", 256 * 1024 * 1024),
    ("grpc.max_send_message_length", 256 * 1024 * 1024),
]


def create_channel(addr: str) -> grpc.Channel:
    """Create gRPC channel with large message support."""
    return grpc.insecure_channel(addr, options=_GRPC_OPTIONS)


@app.command()
def health(
    port: int = 50051,
    ip: str = "127.0.0.1",
    timeout: int = 5,
):
    """Test gRPC health check endpoint."""
    addr = f"{ip}:{port}"
    channel = create_channel(addr)
    stub = health_pb2_grpc.HealthStub(channel)

    try:
        request = health_pb2.HealthCheckRequest()
        response = stub.Check(request, timeout=timeout)
        status = health_pb2.HealthCheckResponse.ServingStatus.Name(response.status)
        print(f"Health check: {status}")
        return response.status == health_pb2.HealthCheckResponse.SERVING
    except grpc.RpcError as e:
        print(f"Health check failed: {e.code()} - {e.details()}")
        raise SystemExit(1)


@app.command()
def detection(
    port: int = 50051,
    ip: str = "127.0.0.1",
    image_path: Path = Path(__file__).parent.parent / "fixtures" / "test_image.png",
    output: Path = None,
    token: str = "",
    timeout: int = 30,
    json_output: bool = False,
):
    """Test ObjectDetection.RunDetection."""
    addr = f"{ip}:{port}"
    metadata = (("authorization", f"Bearer {token}"),) if token else ()

    # Load image
    image = imageio.imread(str(image_path))
    print(f"Loaded image: {image.shape}, dtype={image.dtype}")

    # Create request
    image_data = serialize_from_numpy_to_image_data(image)
    request = proto.DetectionRequest(
        image_data=image_data,
        detection_settings=proto.DetectionSettings(scaling_hint=1.0),
    )

    # Call service
    channel = create_channel(addr)
    stub = proto.ObjectDetectionStub(channel)

    start_time = time.perf_counter()
    try:
        response = stub.RunDetection(request, metadata=metadata, timeout=timeout)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if json_output:
            result = {
                "detections": len(response.detections),
                "elapsed_ms": elapsed_ms,
                "success": True,
            }
            print(json.dumps(result))
        else:
            print(f"Detected {len(response.detections)} objects in {elapsed_ms:.1f}ms")

            if output:
                # Save detection polygons as JSON
                polygons = []
                for det in response.detections:
                    if det.roi.HasField("polygon"):
                        pts = [(p.x, p.y) for p in det.roi.polygon.points]
                        polygons.append({"score": det.score, "polygon": pts})
                with open(output, "w") as f:
                    json.dump(polygons, f, indent=2)
                print(f"Saved detections to {output}")

        return len(response.detections)

    except grpc.RpcError as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if json_output:
            result = {
                "success": False,
                "error": str(e.details()),
                "code": str(e.code()),
                "elapsed_ms": elapsed_ms,
            }
            print(json.dumps(result))
        else:
            print(f"Detection failed: {e.code()} - {e.details()}")
        raise SystemExit(1)


@app.command()
def process(
    port: int = 50051,
    ip: str = "127.0.0.1",
    image_path: Path = Path(__file__).parent.parent / "fixtures" / "test_image.png",
    output: Path = None,
    op_name: str = "",
    token: str = "",
    timeout: int = 30,
    json_output: bool = False,
):
    """Test ProcessImage.Run."""
    addr = f"{ip}:{port}"
    metadata = (("authorization", f"Bearer {token}"),) if token else ()

    # Load image
    image = imageio.imread(str(image_path))
    print(f"Loaded image: {image.shape}, dtype={image.dtype}")

    # Create request
    image_data = serialize_from_numpy_to_image_data(image)
    request = proto.ProcessRequest(
        image_data=image_data,
        op_name=op_name,
    )

    # Call service
    channel = create_channel(addr)
    stub = proto.ProcessImageStub(channel)

    start_time = time.perf_counter()
    try:
        response = stub.Run(request, metadata=metadata, timeout=timeout)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Decode result
        result = deserialize_image_data(response.image_data)

        if json_output:
            result_json = {
                "shape": list(result.shape),
                "dtype": str(result.dtype),
                "elapsed_ms": elapsed_ms,
                "success": True,
                "is_lazy": response.image_data.HasField("lazy_data"),
            }
            print(json.dumps(result_json))
        else:
            lazy_note = " (lazy)" if response.image_data.HasField("lazy_data") else " (eager)"
            print(f"Result: {result.shape}, dtype={result.dtype}{lazy_note} in {elapsed_ms:.1f}ms")

            if output:
                # Save result as image
                if result.ndim == 3 and result.shape[-1] == 1:
                    result = result.squeeze(-1)
                imageio.imwrite(str(output), result.astype(np.uint8))
                print(f"Saved result to {output}")

        return result.shape

    except grpc.RpcError as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if json_output:
            result_json = {
                "success": False,
                "error": str(e.details()),
                "code": str(e.code()),
                "elapsed_ms": elapsed_ms,
            }
            print(json.dumps(result_json))
        else:
            print(f"Process failed: {e.code()} - {e.details()}")
        raise SystemExit(1)


@app.command()
def ops(
    port: int = 50051,
    ip: str = "127.0.0.1",
    token: str = "",
    timeout: int = 10,
):
    """Test GetOpNames endpoint."""
    addr = f"{ip}:{port}"
    metadata = (("authorization", f"Bearer {token}"),) if token else ()

    channel = create_channel(addr)
    stub = proto.ProcessImageStub(channel)

    from google.protobuf.empty_pb2 import Empty

    try:
        response = stub.GetOpNames(Empty(), metadata=metadata, timeout=timeout)
        print(f"Available operations: {list(response.names)}")

        for name in response.names:
            schema = response.op_schemas.get(name)
            if schema:
                print(f"\n{name}:")
                print(f"  Description: {schema.description}")
                if schema.default_kwargs.fields:
                    print(f"  Default kwargs: {dict(schema.default_kwargs.fields)}")

        return list(response.names)

    except grpc.RpcError as e:
        print(f"GetOpNames failed: {e.code()} - {e.details()}")
        raise SystemExit(1)


@app.command()
def streaming(
    port: int = 50051,
    ip: str = "127.0.0.1",
    image_path: Path = Path(__file__).parent.parent / "fixtures" / "test_image.png",
    iterations: int = 4,
    token: str = "",
    timeout: int = 60,
):
    """Test streaming endpoints."""
    addr = f"{ip}:{port}"
    metadata = (("authorization", f"Bearer {token}"),) if token else ()

    image = imageio.imread(str(image_path))
    print(f"Testing streaming with {iterations} iterations")

    def request_generator():
        image_data = serialize_from_numpy_to_image_data(image)
        yield proto.ProcessRequest(image_data=image_data)
        for _ in range(iterations - 1):
            yield proto.ProcessRequest()

    channel = create_channel(addr)
    stub = proto.ProcessImageStub(channel)

    start_time = time.perf_counter()
    response_count = 0

    try:
        for response in stub.RunStream(request_generator(), metadata=metadata, timeout=timeout):
            response_count += 1
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            result = deserialize_image_data(response.image_data)
            print(f"  Response {response_count}: shape={result.shape} at {elapsed_ms:.1f}ms")

        print(f"\nStreaming test passed: {response_count} responses")

    except grpc.RpcError as e:
        print(f"Streaming test failed: {e.code()} - {e.details()}")
        raise SystemExit(1)


@app.command()
def all_tests(
    port: int = 50051,
    ip: str = "127.0.0.1",
    image_path: Path = Path(__file__).parent.parent / "fixtures" / "test_image.png",
    token: str = "",
):
    """Run all tests against a service."""
    print(f"Testing service at {ip}:{port}")
    print("=" * 50)

    results = []

    # Health check
    print("\n1. Health check...")
    try:
        health(port, ip)
        results.append(("health", True))
    except SystemExit:
        results.append(("health", False))

    # Detection
    print("\n2. Detection...")
    try:
        detection(port, ip, image_path, token=token)
        results.append(("detection", True))
    except SystemExit:
        results.append(("detection", False))

    # Process
    print("\n3. Process...")
    try:
        process(port, ip, image_path, token=token)
        results.append(("process", True))
    except SystemExit:
        results.append(("process", False))

    # GetOpNames
    print("\n4. GetOpNames...")
    try:
        ops(port, ip, token=token)
        results.append(("ops", True))
    except SystemExit:
        results.append(("ops", False))

    # Summary
    print("\n" + "=" * 50)
    print("Results:")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    app()