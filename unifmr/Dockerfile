FROM jiyuuchc/biopb-image-base:6c6053a

WORKDIR $HOME

# Install Python dependencies (torch from the CUDA 12.8 wheel index).
COPY --chown=biopb:biopb requirements.txt ./
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt

# Optional: build for CPUs without SSE4.2/AVX (e.g. old AMD Opterons). See
# ../cellpose/BUILD_NO_SSE42.md. pyarrow's wheels are compiled with an SSE4.2
# baseline and SIGILL on import on such CPUs; it is only needed for the
# lazy/Flight side channel (Phase 2), which run_server disables at runtime when
# pyarrow is absent. Removing it lets pandas/dask and biopb degrade gracefully;
# eager image data (Phase 1) is unaffected. Build with: --build-arg NO_SSE42=1
# NOTE: pyarrow is installed root-owned in /usr/local/lib; this stage runs as the
# unprivileged `biopb` user, so the uninstall must switch to root or it fails
# silently with PermissionError and pyarrow stays in the image.
ARG NO_SSE42=0
USER root
RUN if [ "$NO_SSE42" = "1" ]; then pip uninstall -y pyarrow || true; fi
USER biopb

# Download the finetuned UNiFMIR checkpoints (all task heads, ~416 MB) and place
# the directory holding the SwinIR* checkpoint folders at ./experiment, matching
# --ckpt-dir below. The find/mv makes this robust to the tarball's internal root.
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

ENTRYPOINT ["python", "./unifmr_server.py"]
CMD ["--no-token", "--ckpt-dir", "./experiment"]
