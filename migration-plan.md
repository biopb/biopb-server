# Migration Plan: biopb-base → jiyuuchc/biopb-image-base

## Overview

This plan outlines the migration from the local `biopb-base/` directory to using the external `jiyuuchc/biopb-image-base` Docker image from the `biopb` repository.

---

## Current State

| Aspect | Current (biopb-base/) | Target (biopb-image-base) |
|--------|----------------------|---------------------------|
| **Build location** | Local Dockerfile in repo | External repo (`../biopb/biopb-image-server/`) |
| **Image name** | `jiyuuchc/biopb-base:x.y.z-N` | `jiyuuchc/biopb-image-base:sha` |
| **Versioning** | VERSION file (`0.2.5-1`) | Git SHA + latest tag |
| **Utilities path** | `/opt/biopb/common.py` etc. | `/opt/biopb/biopb_image_base/` package |
| **Import style** | `from common import ...` | `from biopb_image_base import ...` |
| **Image protocol** | `Pixels` field only | `ImageData` (eager/lazy/pixels) |
| **Tensor cache** | Not available | Embedded tensor server support |

---

## API Changes

### Import Migration

```python
# OLD
from common import decode_image, encode_image, BiopbServicerBase, setup_logging, parse_kwargs, validate_kwargs
from server import run_server

# NEW
from biopb_image_base import decode_image_data, encode_image, BiopbServicerBase, setup_logging, run_server
from utils import parse_kwargs, validate_kwargs  # Local utils for functions not in new base
```

**Note**: `parse_kwargs` and `validate_kwargs` are not exported from `biopb_image_base`. A local `utils.py` was created to provide these functions for services that need them (cellpose, ucell).

### Function Changes

| Old Function | New Function | Notes |
|--------------|--------------|-------|
| `decode_image(pixels)` | `decode_image_data(image_data)` | Accepts `ImageData`, handles eager/lazy data |
| `encode_image(array)` → `Pixels` | `encode_image(array)` → `ImageData` | Returns `ImageData` with `eager_data` |
| `BiopbServicerBase(use_lock)` | `BiopbServicerBase(use_lock, tensor_cache)` | Optional tensor_cache for lazy results |

### Response Construction

```python
# OLD
response = proto.ProcessResponse(
    image_data = proto.ImageData(pixels = encode_image(mask)),
)

# NEW
response = proto.ProcessResponse(
    image_data = encode_image(mask),  # Returns ImageData directly
)

# NEW with lazy support (optional)
response = proto.ProcessResponse(
    image_data = return_lazy_or_eager(result, self._tensor_cache),
)
```

### Request Parsing

```python
# OLD
pixels = request.image_data.pixels
image = decode_image(pixels)

# NEW (backward compatible)
image = decode_image_data(request.image_data)  # Works with all formats
```

---

## Services to Migrate

| Service | Status | Notes |
|---------|--------|-------|
| `cellpose/` | Active | Migrate |
| `cellpose-sam/` | Active | Migrate |
| `lacss/` | Active | Migrate |
| `samcell/` | Active | Migrate |
| `ucell/` | Active | Migrate |
| `ctxseg/` | **Deprecated** | No migration needed |
| `ctxseg-d/` | **Deprecated** | No migration needed |
| `osilab/` | **Deprecated** | No migration needed |

---

## Migration Phases

### Phase 0: Set Up Testing Framework (Recommended First)

Create the testing infrastructure before migration to enable verification at each step:

1. Create `tests/` directory structure
2. Add `tests/conftest.py` with Docker service fixtures
3. Add `tests/test_client.py` CLI
4. Add smoke tests for current services (using old biopb-base)
5. Add `pyproject.toml` or `tests/requirements.txt`
6. Run smoke tests against current services to verify baseline

This phase ensures:
- Tests pass against current services before migration
- Migration changes can be verified immediately
- Future development has a testing baseline

### Phase 1: Update Base Image Reference (Low Risk)

Update all active service Dockerfiles to use `jiyuuchc/biopb-image-base`:

```dockerfile
# OLD
FROM jiyuuchc/biopb-base:0.2.5-1

# NEW
FROM jiyuuchc/biopb-image-base:<version>
```

**Version strategy**: Use a specific SHA tag from the biopb repo build, or `latest` for development.

**Affected files** (5 services):
- `cellpose/Dockerfile`
- `cellpose-sam/Dockerfile`
- `lacss/Dockerfile`
- `samcell/Dockerfile`
- `ucell/Dockerfile`

### Phase 2: Update Import Statements (Medium Risk)

For each active service server file, update imports:

```python
# Replace
from common import decode_image, encode_image, BiopbServicerBase, setup_logging, parse_kwargs, validate_kwargs
from server import run_server

# With
from biopb_image_base import decode_image_data, encode_image, BiopbServicerBase, setup_logging, parse_kwargs
from biopb_image_base import validate_kwargs, run_server
```

**Affected files** (5 files):
- `cellpose/cellpose_server.py`
- `cellpose-sam/cellpose_server.py`
- `lacss/lacss_server.py`
- `samcell/samcell_server.py`
- `ucell/ucell_server.py`

### Phase 3: Update decode_image → decode_image_data (Medium Risk)

Replace all `decode_image()` calls:

```python
# OLD
pixels = request.image_data.pixels
image = decode_image(pixels)

# NEW
image = decode_image_data(request.image_data)
```

This change maintains backward compatibility because:
- `ImageData.pixels` is still supported by `decode_image_data()`
- The new function also handles `eager_data` and `lazy_data`

### Phase 4: Update Response Construction (Low Risk)

For `ProcessResponse`, simplify the construction:

```python
# OLD
response = proto.ProcessResponse(
    image_data = proto.ImageData(pixels = encode_image(mask)),
)

# NEW
response = proto.ProcessResponse(
    image_data = encode_image(mask),
)
```

### Phase 5: Add Health Check Probe (Optional Enhancement)

The new base image includes `grpc_health_probe`. Add health checks to derived images:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD grpc_health_probe -addr=localhost:50051 || exit 1
```

### Phase 6: Remove biopb-base Directory (Cleanup)

After all services are migrated and tested:

1. Remove `biopb-base/` directory
2. Remove deprecated service directories: `ctxseg/`, `ctxseg-d/`, `osilab/`
3. Update CLAUDE.md to reflect new architecture
4. Update GitHub Actions workflow if needed

---

## Version Pinning Strategy

### Option A: SHA Pinning (Recommended for Production)

```dockerfile
FROM jiyuuchc/biopb-image-base:abc1234
```

- Immutable, reproducible builds
- SHA from biopb repo: `git rev-parse --short HEAD`

### Option B: Latest Tag (Development Only)

```dockerfile
FROM jiyuuchc/biopb-image-base:latest
```

### Option C: VERSION File with Build Number

Create a `BASE_VERSION` file in each service:

```dockerfile
ARG BASE_VERSION=latest
FROM jiyuuchc/biopb-image-base:${BASE_VERSION}
```

---

## Testing Framework

### Current State

The repository has no testing framework. Tests are ad-hoc manual runs using `biopb-base/test.py`.

### Proposed Architecture

```
tests/
├── conftest.py              # Shared fixtures (Docker service launcher, test client)
├── test_client.py           # Reusable gRPC test client CLI
├── test_service_base.py     # Base test class for all services
├── services/
│   ├── test_cellpose.py     # Cellpose-specific tests
│   ├── test_cellpose_sam.py # Cellpose-SAM-specific tests
│   ├── test_lacss.py        # Lacss-specific tests
│   ├── test_samcell.py      # Samcell-specific tests
│   └── test_ucell.py        # Ucell-specific tests
├── fixtures/
│   └── test_image.png       # Shared test images
│   └── test_image_3d.tif    # 3D test image (for services supporting 3D)
└── utils/
    └── image_utils.py       # Image generation/loading helpers
```

### Test Categories

#### 1. Smoke Tests (Required for All Services)

Basic connectivity and functionality tests:

```python
class TestServiceSmoke:
    """Minimal tests to verify service is running correctly."""

    def test_health_check(self, service_channel):
        """gRPC health endpoint returns SERVING."""

    def test_detection_returns_results(self, service_channel, test_image):
        """RunDetection returns valid DetectionResponse."""

    def test_detection_empty_image(self, service_channel):
        """Service handles edge case gracefully."""

    def test_get_op_names(self, service_channel):
        """GetOpNames returns valid operation names."""
```

#### 2. Integration Tests (Service-Specific)

Tests specific to each service's capabilities:

```python
# tests/services/test_cellpose.py
class TestCellposeIntegration:
    """Cellpose-specific integration tests."""

    def test_2d_detection(self, service_channel, test_image_2d):
        """2D cell segmentation works."""

    def test_2d_process_mask(self, service_channel, test_image_2d):
        """ProcessImage returns segmentation mask."""

    def test_kwargs_diameter(self, service_channel, test_image_2d):
        """Custom diameter parameter works."""

    def test_kwargs_channels(self, service_channel, test_image_multichannel):
        """Channel specification works."""

    def test_3d_raises_error(self, service_channel, test_image_3d):
        """3D input raises appropriate error for RunDetection."""
```

#### 3. Contract Tests (API Compliance)

Verify service implements biopb.image protocol correctly:

```python
class TestServiceContract:
    """Verify biopb.image protocol compliance."""

    def test_image_data_eager_format(self, service_channel):
        """Accepts eager_data ImageData format."""

    def test_image_data_pixels_format(self, service_channel):
        """Accepts legacy pixels format (backward compat)."""

    def test_response_has_required_fields(self, service_channel):
        """DetectionResponse has required field structure."""

    def test_kwargs_validation(self, service_channel):
        """Invalid kwargs return INVALID_ARGUMENT status."""
```

### Fixture Design

```python
# tests/conftest.py

import subprocess
import time
import tempfile
import pytest
import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

@pytest.fixture(scope="session")
def docker_service(request):
    """Launch service Docker container for testing.

    Usage:
        @pytest.mark.parametrize("service", ["cellpose", "lacss"])
        def test_x(docker_service):
            channel = docker_service.channel
    """
    service_name = request.param
    container_name = f"biopb-test-{service_name}"

    # Build or pull image
    subprocess.run([
        "docker", "build", "-t", f"{service_name}-test",
        f"./{service_name}"
    ], check=True)

    # Run container
    proc = subprocess.Popen([
        "docker", "run", "--rm", "--gpus=all",
        "--name", container_name,
        "-p", "50051:50051",
        f"{service_name}-test",
        "--no-token", "--debug"
    ])

    # Wait for health check
    server_addr = "127.0.0.1:50051"
    for _ in range(30):
        try:
            channel = grpc.insecure_channel(server_addr)
            stub = health_pb2_grpc.HealthStub(channel)
            stub.Check(health_pb2.HealthCheckRequest(), timeout=2)
            break
        except:
            time.sleep(1)

    yield type("ServiceHandle", (), {"channel": channel, "addr": server_addr})

    # Cleanup
    subprocess.run(["docker", "stop", container_name], check=False)
    proc.wait()

@pytest.fixture
def test_image_2d():
    """Standard 2D test image."""
    import imageio
    return imageio.imread("tests/fixtures/test_image.png")

@pytest.fixture
def test_image_multichannel():
    """Multi-channel fluorescence test image."""
    import numpy as np
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
```

### Test Runner Options

#### Local Testing (Docker)

```bash
# Run all tests against Docker containers
pytest tests/ -v --requires-gpu

# Run specific service tests
pytest tests/services/test_cellpose.py -v

# Run smoke tests only
pytest tests/ -v -k "smoke"
```

#### CI Testing (GitHub Actions)

```yaml
# .github/workflows/test.yml
name: Service Tests

on:
  pull_request:
  workflow_dispatch:

jobs:
  test:
    runs-on: [self-hosted, gpu]  # Requires GPU runner
    strategy:
      matrix:
        service: [cellpose, cellpose-sam, lacss, samcell, ucell]
    steps:
      - uses: actions/checkout@v4

      - name: Build service image
        run: docker build -t ${{ matrix.service }}-test ./${{ matrix.service }}

      - name: Run tests
        run: |
          docker run --rm --gpus=all -p 50051:50051 \
            ${{ matrix.service }}-test --no-token &
          sleep 10
          pytest tests/services/test_${{ matrix.service }}.py -v
```

### Test Client CLI

Standalone CLI for manual testing:

```python
# tests/test_client.py
"""CLI test client for biopb.image services."""

import typer
import grpc
import imageio
import biopb.image as proto

app = typer.Typer()

@app.command()
def detection(
    port: int = 50051,
    image_path: str = "tests/fixtures/test_image.png",
    output: str = None,
    token: str = None,
):
    """Test ObjectDetection.RunDetection."""
    # ... implementation

@app.command()
def process(
    port: int = 50051,
    image_path: str,
    output: str,
    op_name: str = None,
):
    """Test ProcessImage.Run."""
    # ... implementation

@app.command()
def health(port: int = 50051):
    """Test gRPC health check."""
    # ... implementation

@app.command()
def ops(port: int = 50051):
    """Test GetOpNames."""
    # ... implementation
```

### pytest Configuration

```toml
# pyproject.toml (add to existing or create new)
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = [
    "smoke: basic connectivity tests",
    "integration: service-specific tests",
    "contract: API compliance tests",
    "requires_gpu: tests that need GPU",
]
```

### Dependencies

```txt
# tests/requirements.txt (or add to pyproject.toml)
pytest >= 7.0
pytest-timeout
grpcio
grpcio-health-checking
biopb  # from biopb repo
numpy
imageio
typer
```

---

## Testing Checklist

### Pre-Migration (Phase 0)

1. Verify pytest runs: `pytest tests/ -v --collect-only`
2. Run smoke tests against current services: `pytest tests/ -v -k "smoke"`
3. Document baseline behavior (detection counts, processing times)

### For Each Migrated Service (5 total):

1. **Build test**: `docker build -t <service>-test .`
2. **Run test**: `docker run --gpus=all -p 50051:50051 <service>-test --no-token --debug`
3. **Smoke tests**: `pytest tests/services/test_<service>.py -v -k "smoke"`
4. **Integration tests**: `pytest tests/services/test_<service>.py -v`
5. **Health check**: `grpc_health_probe -addr=localhost:50051` (if added)
6. **Compare baseline**: Verify detection counts/mask shapes match pre-migration

### Post-Migration

1. Run full test suite: `pytest tests/ -v`
2. Verify CI workflow runs successfully (if set up)
3. Update test fixtures if lazy data format differs

---

## Rollback Plan

If issues arise:

1. Revert Dockerfile `FROM` line to old base image version
2. Revert import statements
3. Old `biopb-base/` can be temporarily restored from git history

---

## Timeline Estimate

| Phase | Duration | Services |
|-------|----------|----------|
| Phase 0 (Testing framework) | 1-2 hours | Repository setup |
| Phase 1 (Dockerfiles) | 30 min | 5 services |
| Phase 2 (Imports) | 15 min | 5 files |
| Phase 3 (decode calls) | 15 min | ~10 locations |
| Phase 4 (Responses) | 10 min | ~3 locations |
| Phase 5 (Health checks) | 10 min | Optional |
| Phase 6 (Cleanup) | 10 min | Repository |
| **Testing (verify migration)** | 1-2 hours | 5 services |

**Total**: ~3-4 hours

**Recommended order**: Set up testing framework first (Phase 0), then migrate services with tests verifying each step.

---

## Dependencies

1. `jiyuuchc/biopb-image-base` must be built and pushed to Docker Hub
2. Verify compatible `biopb` package version in base image matches service expectations
3. Ensure all services have GPU/model dependencies compatible with new base Python 3.10

---

## Notes

### Backward Compatibility

The new `decode_image_data()` function is backward compatible with existing `Pixels` requests. Services will automatically support the new `ImageData` protocol (eager/lazy) without code changes.

### Optional Enhancements

Services can optionally adopt lazy data handling by:
1. Passing `tensor_cache` to `BiopbServicerBase`
2. Using `return_lazy_or_eager()` for large results
3. Running with `--cache-dir /data/cache --cache-size 32GB`

This is **not required** for migration but enables future optimization.

### validate_kwargs Location

`validate_kwargs` is available directly from `biopb_image_base` (it's in `common.py` which is exposed via `__init__.py`). No special handling needed.

---

## Migration Testing Strategy

### Baseline Capture

Before migration, run tests against current services and capture:

```bash
# Capture baseline for each service
pytest tests/ -v --tb=no \
    --json-report --json-report-file=baseline_cellpose.json \
    -k "cellpose"
```

Key metrics to compare:
- Detection count on standard test image
- ProcessImage mask shape
- Response latency (approximate)
- Error handling behavior

### Migration Verification

After each service migration:

1. Run same tests against migrated service
2. Compare results to baseline
3. Acceptable differences:
   - Detection count ±10% (ML models may vary slightly)
   - Latency changes (new base may be faster/slower)
4. Unacceptable differences:
   - Empty detection responses
   - Mask shape mismatches
   - New errors/crashes