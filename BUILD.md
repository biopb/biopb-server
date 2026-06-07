# Build Pipeline

This document describes how to build and push Docker images for biopb services.

## Services

Each top-level subdirectory with a `Dockerfile` is an independent, deployable
service. `build.sh` and the CI workflow discover them automatically via
`*/Dockerfile`:

| Service | Description |
|---------|-------------|
| `cellpose/` | Cellpose Cyto3 cell segmentation |
| `cellpose-sam/` | Cellpose-SAM |
| `lacss/` | Lacss3 (JAX-based) |
| `samcell/` | Finetuned SAM model |
| `ucell/` | FRM-based model |
| `unifmir/` | UNiFMIR image restoration |

There is **no base image in this repo.** Every service inherits from
`jiyuuchc/biopb-image-base:<sha>`, which is built and published from the separate
[`biopb/biopb`](https://github.com/biopb/biopb) monorepo (subdirectory
`biopb-image-runtime`). See "Base Image Dependency" below.

## Versioning

Each service has a `VERSION` file that defines the image version tag, e.g.:

```
# cellpose/VERSION
0.3.0
```

If no VERSION file exists, `latest` is used as the version tag.

## Image Naming

Images are tagged as: `<REGISTRY>/<IMAGE_PREFIX>/<service-name>:<tag>`
(defaults `docker.io` / `jiyuuchc`).

Examples:
- `jiyuuchc/cellpose:0.3.0`
- `jiyuuchc/samcell:latest`

## Local Build

Use `build.sh` for local builds with CI-consistent tagging:

### Build Single Service

```bash
# Build cellpose without pushing
./build.sh --no-push cellpose

# Build and push cellpose
./build.sh cellpose
```

### Build Multiple / All Services

```bash
# Build specific services
./build.sh cellpose samcell

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
| `<service>` | Build specific service(s) |
| `all` | Build all services (every dir with a `Dockerfile`) |

### Generated Tags

The script generates tags consistent with CI:

| Tag | Description | Push Only |
|-----|-------------|-----------|
| `<sha>` | Git short commit hash | No |
| `<version>` | From the service's `VERSION` file | Yes |
| `latest` | Latest build | Yes |

Example output for `cellpose` with VERSION `0.3.0` and commit `9449e9a`:
- `jiyuuchc/cellpose:9449e9a` (always)
- `jiyuuchc/cellpose:0.3.0` (push only)
- `jiyuuchc/cellpose:latest` (push only)

### Environment Variables

```bash
# Override registry (default: docker.io)
REGISTRY=ghcr.io ./build.sh cellpose

# Override image prefix (default: jiyuuchc)
IMAGE_PREFIX=myorg ./build.sh cellpose
```

## Remote Build (GitHub Actions)

Builds are triggered manually via the **Docker Build and Push** workflow
(`.github/workflows/docker-build.yml`) using workflow dispatch.

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

## Running a Built Image

```bash
docker run --gpus=all -p 50051:50051 <image> [--local|--no-token|--debug|--token]
```

Requires NVIDIA driver ≥525 and the NVIDIA Container Toolkit.

## Testing

Tests are run with `pytest`, not from any per-service test script. There are two
tiers (see [CLAUDE.md](CLAUDE.md) and [pyproject.toml](pyproject.toml) for full
details):

- **Unit tier** (`tests/unit/`, marker `unit`) — fast, GPU-free, container-free.
  Requires only `pip install -e ".[test]"`.
  ```bash
  .venv/bin/pytest -m unit
  ```
- **Integration tier** (`tests/services/`) — gRPC clients against running
  containers. Requires pre-built images tagged `<service>:test` and a GPU.
  ```bash
  docker build -t cellpose:test cellpose/
  .venv/bin/pytest tests/services/test_cellpose.py
  ```

## Base Image Dependency

Every service's `Dockerfile` starts with:

```dockerfile
FROM jiyuuchc/biopb-image-base:<sha>
```

When the base image changes:

1. Bump the `FROM jiyuuchc/biopb-image-base:<sha>` pin in each service's
   `Dockerfile`.
2. Bump the matching `biopb-image-base @ git+...@<commit>` pin in
   [pyproject.toml](pyproject.toml) so the dev/test venv stays identical to the
   container.
3. Rebuild dependent services (manual trigger required) — the base is not built
   in this repo.

## Build Matrix Example

```bash
# Quick test build (no push)
./build.sh --no-push cellpose

# Build and push a couple of services
./build.sh cellpose samcell
```
