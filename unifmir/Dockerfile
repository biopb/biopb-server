FROM jiyuuchc/biopb-image-base:sha-d38e184

WORKDIR $HOME

# Install Python dependencies (torch from the CUDA 12.8 wheel index). The large
# CUDA runtime wheels (torch/cudnn/cufft, ~200 MB-1 GB each) can arrive truncated
# and fail pip's hash check, so retry. Use --no-cache-dir so a corrupted wheel is
# never reused from cache on the next attempt (which would re-fail indefinitely).
COPY --chown=biopb:biopb requirements.txt ./
RUN ok=; for i in 1 2 3 4 5; do \
        if pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt; then ok=1; break; fi; \
        echo "pip install attempt $i failed; retrying after 10s"; sleep 10; \
    done; \
    [ "$ok" = 1 ]

# Optional: build for CPUs without SSE4.2/AVX (e.g. old AMD Opterons).
# pyarrow's wheels are compiled with an SSE4.2 baseline and SIGILL on import on
# such CPUs, and so does anything that imports it (pandas, dask). The base image
# advertises eager-only when the tensor server can't be imported, but that never
# runs if the import itself kills the process, so the package is removed at build
# time. Eager/pixels image data is unaffected. Build with: --build-arg NO_SSE42=1
# See ../cellpose/BUILD_NO_SSE42.md.
# NOTE: pyarrow is installed root-owned in /usr/local/lib; this stage runs as the
# unprivileged `biopb` user, so the uninstall must switch to root or it fails
# silently with PermissionError and pyarrow stays in the image.
ARG NO_SSE42=0
USER root
RUN if [ "$NO_SSE42" = "1" ]; then pip uninstall -y pyarrow || true; fi
USER biopb

# Download the finetuned UNiFMIR checkpoints (all task heads, ~416 MB) and place
# the directory holding the SwinIR* checkpoint folders at ./experiment, matching
# _CKPT_DIR in unifmir_server.py. The find/mv makes this robust to the tarball's
# internal root.
RUN set -e; \
    wget -q https://github.com/cxm12/UNiFMIR/releases/download/2023.10.05/model.tgz -O /tmp/model.tgz; \
    mkdir -p /tmp/ckpt; tar -xzf /tmp/model.tgz -C /tmp/ckpt; rm /tmp/model.tgz; \
    src="$(dirname "$(find /tmp/ckpt -type d -name 'SwinIRF-actin' | head -1)")"; \
    test -n "$src"; mv "$src" "$HOME/experiment"; rm -rf /tmp/ckpt

# Copy service code (includes the vendored model/ package).
COPY --chown=biopb:biopb . .

EXPOSE 50051

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD grpc_health_probe -addr=localhost:50051 || exit 1

# Off loopback the server requires a token: $BIOPB_ALGORITHM_TOKEN, or one it
# mints and prints. Bound to 127.0.0.1 (with --network host) it takes none.
ENTRYPOINT ["python", "./unifmir_server.py"]
CMD ["--host", "0.0.0.0"]
