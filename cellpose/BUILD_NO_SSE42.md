# Building cellpose for old CPUs without SSE4.2 / AVX

## Problem

On old AMD CPUs (verified on an **AMD Opteron 6172**, Magny-Cours, ~2010) the
cellpose container crashes on startup with exit code **132 (SIGILL — illegal
instruction)**. The CPU has only `sse, sse2, sse3, sse4a, popcnt` — **no
SSE4.1, no SSE4.2, no AVX**.

The culprit is **not AVX**: it is **`pyarrow`**. Apache Arrow wheels are compiled
with a fixed `ARROW_SIMD_LEVEL=SSE4_2` baseline (not behind runtime dispatch),
so `import pyarrow` executes an SSE4.2 instruction unconditionally and the
process dies with SIGILL. `torch` survives because it does true runtime CPU
feature dispatch down to a generic baseline. There is **no env var** to lower
pyarrow's compiled baseline (`ARROW_USER_SIMD_LEVEL` only caps the *runtime*
dispatch level).

pyarrow is pulled in eagerly through **two independent chains**:

1. `biopb.tensor.client` → `import pyarrow` (only needed for the lazy/Flight
   `TensorFlightClient` data path).
2. `import dask.array` → `import pandas` → `pandas/compat/pyarrow.py: import pyarrow`.
   pandas wraps this in `try/except ImportError`, but **SIGILL is a signal, not
   an exception**, so it cannot be caught — it kills the process.

The eager/pixels image-data path itself never *uses* pyarrow; it just gets
imported at module load.

## Path A — lazy imports + drop pyarrow (implemented)

This is the lightweight fix. cellpose runs on the old CPU for **eager / pixels**
image data (the normal case). Lazy/Flight (`lazy_data`) input is not supported on
such hardware — which is fine, since that path needs Arrow anyway.

Two parts, both required:

### 1. biopb source (ships in `biopb-image-base`)

Make the pyarrow imports lazy so `import biopb.*` no longer pulls pyarrow at
module load (changes in the `biopb` repo, `../biopb`):

- `src/main/python/biopb/tensor/__init__.py` — export `TensorFlightClient` /
  `make_debug_serialized_tensor` lazily via PEP 562 module `__getattr__`.
- `src/main/python/biopb/image/utils.py` — move
  `from biopb.tensor.client import TensorFlightClient` out of the module top and
  into the `lazy_data` branch of `deserialize_image_data`.

Then **rebuild and push `jiyuuchc/biopb-image-base`** and bump the `FROM` tag in
`cellpose/Dockerfile` to the new digest.

### 2. cellpose image — drop pyarrow

`cellpose/Dockerfile` has an opt-in build arg (default off, so normal AVX builds
keep full lazy/Flight support):

```dockerfile
ARG NO_SSE42=0
RUN if [ "$NO_SSE42" = "1" ]; then pip uninstall -y pyarrow || true; fi
```

Build for the old CPU:

```bash
cd cellpose
docker buildx build --build-arg NO_SSE42=1 -t jiyuuchc/cellpose:nosse42 .
```

With pyarrow gone, pandas/dask hit the normal `ImportError` and degrade
gracefully; biopb only attempts to import pyarrow if an actual `lazy_data`
request arrives (which this CPU cannot serve).

## Runtime requirement: must run on GPU (CPU inference SIGFPEs)

This CPU **cannot run cellpose inference on the CPU**: torch's bundled CPU BLAS
(MKL) raises **SIGFPE (exit 136)** on even a trivial matmul here. `import torch`
succeeds (hence the "torch works" impression), but any CPU GEMM crashes. So the
container must do inference on the **GPU**.

The host has a Tesla T4 but needs the CUDA **forward-compat** libraries, so run
with:

```bash
docker run --gpus=all \
  --volume /usr/local/cuda-12.6/compat:/compat \
  -e LD_LIBRARY_PATH=/compat \
  -p 50051:50051 jiyuuchc/cellpose:nosse42 --no-token
```

(Do **not** pass `--no-gpu`/`gpu=False` on this hardware — it forces the MKL CPU
path and crashes.)

### Verification performed

On `jyu@biopb.org` (Opteron 6172 + Tesla T4), in a container built FROM the
prebuilt image with both Path-A parts applied and the compat layer mounted:

- `import dask.array`, `import biopb_image_base`, `from cellpose import models`
  — all import cleanly (previously SIGILL via pyarrow).
- The server boots, `grpc_health_probe` returns `SERVING`, and a real
  `RunDetection` on an eager image runs on the GPU (`cuda: True, Tesla T4`) and
  returns the correct detections.
- With `--cache-dir` but pyarrow absent, `run_server` logs a warning, **does not**
  start the tensor side channel, and the server stays healthy (eager-only).
- Fixed an unrelated pre-existing bug along the way: `RunDetection` was passing
  the default `async_result` kwarg into `CellposeModel.eval()` (it is a
  lazy-output control flag, ProcessImage-only) — now stripped in `process_input`.

## Path B — build a pyarrow without the SSE4.2 baseline (for full lazy/Flight)

Only needed if you want the **lazy/Flight tensor pipeline** to work on these old
CPUs too. Heavier: a custom Arrow C++ + Python build.

### Plan

1. **Build env.** In a builder stage (matching the base image's Python 3.10 /
   manylinux toolchain), install: `cmake`, `ninja`, a C++17 compiler, plus
   Arrow's bundled-dependency build deps. Easiest is to let Arrow vendor its
   thirdparty deps (`ARROW_DEPENDENCY_SOURCE=BUNDLED`).

2. **Build Arrow C++ with SIMD disabled** — the key flags:
   ```
   -DARROW_SIMD_LEVEL=NONE
   -DARROW_RUNTIME_SIMD_LEVEL=NONE
   ```
   Enable the features the Flight client needs and nothing heavy you don't:
   ```
   -DARROW_FLIGHT=ON -DARROW_IPC=ON -DARROW_COMPUTE=ON
   -DARROW_DATASET=OFF -DARROW_PARQUET=OFF -DARROW_CSV=OFF
   -DARROW_WITH_SNAPPY=ON -DARROW_WITH_ZSTD=ON
   -DCMAKE_BUILD_TYPE=Release
   ```
   Also pass `-march`/`-mtune` no higher than the target (e.g. `-march=barcelona`
   / `amdfam10`, or simply `-mno-sse4.1 -mno-sse4.2 -mno-avx`) so the compiler
   never emits SSE4.2+ in Arrow's own code.

3. **Build the pyarrow wheel** from `python/` against that Arrow with
   `PYARROW_WITH_FLIGHT=1 PYARROW_WITH_DATASET=0 PYARROW_WITH_PARQUET=0`, then
   `pip wheel`/`python setup.py bdist_wheel`. **Pin the Arrow version to match
   what `biopb-tensor-server` expects** (`pyarrow>=14.0.0`; the prebuilt image
   used 24.0.0 — pick one version and pin it everywhere).

4. **Ship the wheel.** Either build it in a multi-stage Dockerfile and `COPY`
   the `.whl` into the base image, or host it and `pip install` it. Make this the
   pyarrow that the base image installs for the no-SSE4.2 variant.

5. **Verify on the Opteron**: `python -c "import pyarrow; import pyarrow.flight"`
   must not SIGILL, then exercise a real `lazy_data` round-trip.

### Notes / risks

- Build time is significant (Arrow C++ + bindings; tens of minutes), and you take
  on maintaining a custom wheel pinned to a specific Arrow version.
- `-DARROW_SIMD_LEVEL=NONE` disables SIMD kernels entirely → slower Arrow compute,
  but correctness is fine and the Flight transfer path is not SIMD-bound.
- Consider building it once, caching the wheel artifact, and reusing it across
  base-image rebuilds rather than rebuilding Arrow every time.
- conda-forge / PyPI wheels do **not** help — they all use the SSE4.2 baseline.
