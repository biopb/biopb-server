"""
Server creation helper for biopb services.

Provides a simplified interface for creating gRPC servers with
health checks, interceptors, and standard configuration.
"""

import logging
import secrets
from concurrent import futures
from typing import Optional

import biopb.image as proto
import grpc

from common import TokenValidationInterceptor, _MAX_MSG_SIZE
from health import HealthServicer, add_health_servicer

logger = logging.getLogger(__name__)


def create_server(
    servicer,
    port: int = 50051,
    workers: int = 10,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: Optional[bool] = None,
    debug: bool = False,
    compression: bool = True,
    health_check: bool = True,
    readiness_check: Optional[callable] = None,
) -> tuple[grpc.Server, Optional[str], Optional[HealthServicer]]:
    """
    Create a configured gRPC server with standard features.

    This creates a gRPC server with:
    - ObjectDetection and ProcessImage services registered
    - Optional token authentication
    - Health check service (standard grpc.health.v1.Health)
    - Configurable compression
    - Proper message size limits

    Args:
        servicer: The main servicer implementing ObjectDetection and ProcessImage
        port: Port to listen on (default 50051)
        workers: Thread pool size (default 10)
        ip: IP to bind to (default "0.0.0.0")
        local: Use local server credentials for secure local-only access
        token: Enable token authentication (None = auto based on local flag)
        debug: Enable debug logging
        compression: Enable gzip compression
        health_check: Enable gRPC health check service
        readiness_check: Optional callable for readiness probe

    Returns:
        Tuple of (server, token_string, health_servicer)
        - server: The configured gRPC server (not started)
        - token_string: The auth token if enabled, None otherwise
        - health_servicer: Health servicer for status updates, None if disabled
    """
    # Determine token setting
    if token is None:
        token = not local

    # Generate token if needed
    token_str = None
    if token:
        token_str = secrets.token_urlsafe(64)
        print()
        print("COPY THE TOKEN BELOW FOR ACCESS.")
        print("=======================================================================")
        print(f"{token_str}")
        print("=======================================================================")
        print()

    # Create server with interceptors
    interceptors = [TokenValidationInterceptor(token_str)]

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers),
        compression=grpc.Compression.Gzip if compression else grpc.Compression.NoCompression,
        interceptors=tuple(interceptors),
        options=(
            ("grpc.max_receive_message_length", _MAX_MSG_SIZE),
            ("grpc.max_send_message_length", _MAX_MSG_SIZE),
        ),
    )

    # Register main services
    proto.add_ObjectDetectionServicer_to_server(servicer, server)
    proto.add_ProcessImageServicer_to_server(servicer, server)

    # Register health check service
    health_servicer = None
    if health_check:
        health_servicer = add_health_servicer(server, readiness_check)

    # Add port
    if local:
        server.add_secure_port(f"127.0.0.1:{port}", grpc.local_server_credentials())
        logger.info(f"Server configured with local credentials on 127.0.0.1:{port}")
    else:
        server.add_insecure_port(f"{ip}:{port}")
        logger.info(f"Server configured on {ip}:{port}")

    return server, token_str, health_servicer


def run_server(
    servicer,
    port: int = 50051,
    workers: int = 10,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: Optional[bool] = None,
    debug: bool = False,
    compression: bool = True,
    health_check: bool = True,
    readiness_check: Optional[callable] = None,
) -> None:
    """
    Create and run a gRPC server (blocking).

    This is a convenience function that creates the server and waits for termination.

    Args:
        servicer: The main servicer implementing ObjectDetection and ProcessImage
        port: Port to listen on (default 50051)
        workers: Thread pool size (default 10)
        ip: IP to bind to (default "0.0.0.0")
        local: Use local server credentials for secure local-only access
        token: Enable token authentication (None = auto based on local flag)
        debug: Enable debug logging
        compression: Enable gzip compression
        health_check: Enable gRPC health check service
        readiness_check: Optional callable for readiness probe
    """
    print("server starting ...")

    server, token_str, health_servicer = create_server(
        servicer=servicer,
        port=port,
        workers=workers,
        ip=ip,
        local=local,
        token=token,
        debug=debug,
        compression=compression,
        health_check=health_check,
        readiness_check=readiness_check,
    )

    print("server starting ... ready")

    server.start()
    server.wait_for_termination()