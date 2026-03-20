# Build Pipeline

This document describes how to build and push Docker images for biopb services.

## Services

Each subdirectory is an independent service with its own Dockerfile:

| Service | Description |
|---------|-------------|
| `biopb-base/` | Base image with common utilities (other services inherit from this) |
| `cellpose/` | Cellpose Cyto3 cell segmentation |
| ... |

## Versioning

Each service has a `VERSION` file that defines the image version tag:

```
# biopb-base/VERSION
0.2.1
```

If no VERSION file exists, `latest` is used.

## Image Naming

Images are tagged as: `jiyuuchc/<service-name>:<tag>`

Examples:
- `jiyuuchc/biopb-base:0.2.1`
- `jiyuuchc/cellpose:1.0.0`
- `jiyuuchc/samcell:latest`

## Local Build

Use `build.sh` for local builds with CI-consistent tagging:

### Build Single Service

```bash
# Build biopb-base (no push)
./build.sh --no-push biopb-base

# Build and push cellpose
./build.sh cellpose
```

### Build All Services

```bash
# Build all services without pushing
./build.sh --no-push all

# Build and push all services
./build.sh all
```

### Options

| Option | Description |
|--------|-------------|
| `--no-push` | Build without pushing to registry |
| `--push` | Build and push to registry (default) |
| `<service>` | Build specific service |
| `all` | Build all services |

### Generated Tags

The script generates tags consistent with CI:

| Tag | Description | Push Only |
|-----|-------------|-----------|
| `<sha>` | Git short commit hash | No |
| `<version>` | From VERSION file | Yes |
| `latest` | Latest build | Yes |

Example output for `biopb-base` with VERSION `0.2.1` and commit `9449e9a`:
- `jiyuuchc/biopb-base:9449e9a` (always)
- `jiyuuchc/biopb-base:0.2.1` (push only)
- `jiyuuchc/biopb-base:latest` (push only)

### Environment Variables

```bash
# Override registry (default: docker.io)
REGISTRY=ghcr.io ./build.sh biopb-base

# Override image prefix (default: jiyuuchc)
IMAGE_PREFIX=myorg ./build.sh biopb-base
```

## Remote Build (GitHub Actions)

Builds are triggered manually via GitHub Actions workflow dispatch.

### Trigger Build

1. Go to **Actions** → **Docker Build and Push**
2. Click **Run workflow**
3. Configure:
   - **Services**: Comma-separated list (e.g., `cellpose,lacss`) or `all`
   - **Push to registry**: Enable to push, disable for dry-run

### Workflow Inputs

| Input | Description | Default |
|-------|-------------|---------|
| `services` | Services to build (comma-separated or "all") | `all` |
| `push` | Push images to Docker Hub | `true` |

### Requirements

- Docker Hub credentials must be configured in repository secrets:
  - `DOCKERHUB_USERNAME`
  - `DOCKERHUB_TOKEN`

## Testing

Test scripts are available in `biopb-base/`:

```bash
# Test unary RPC (RunDetection, Run)
python biopb-base/test.py --port 50051 --token <token>

# Test streaming RPC (RunStream)
python biopb-base/test_streaming.py --port 50051 --token <token>
```

### Test Options

| Option | Description | Default |
|--------|-------------|---------|
| `--port` | Server port | 50051 |
| `--ip` | Server IP | 127.0.0.1 |
| `--token` | Auth token | (empty) |
| `--debug` | Enable debug logging | False |
| `--health` | Test health check first | True |

## Base Image Dependency

Services that inherit from `biopb-base` should:

1. Use the versioned tag in their Dockerfile:
   ```dockerfile
   FROM jiyuuchc/biopb-base:0.2.1
   ```

2. Rebuild when base image changes (manual trigger required)

## Build Matrix Examples

```bash
# Build base image first, then dependent services
./build.sh biopb-base
./build.sh cellpose samcell

# Quick test build (no push)
./build.sh --no-push cellpose
```