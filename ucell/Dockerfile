FROM jiyuuchc/biopb-image-base:ec58ca5

# Install git for pip install from git repos
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
USER biopb

WORKDIR $HOME

# Install Python dependencies
COPY --chown=biopb:biopb requirements.txt ./
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt

RUN pip install --no-cache-dir --ignore-requires-python git+https://github.com/jiyuuchc/ucell.git

# Optional: build for CPUs without SSE4.2/AVX (e.g. old AMD Opterons). See
# ../cellpose/BUILD_NO_SSE42.md. pyarrow's wheels are compiled with an SSE4.2
# baseline and SIGILL on import on such CPUs; it is only needed for the
# lazy/Flight side channel, which run_server disables at runtime when pyarrow is
# absent. Removing it lets pandas/dask and biopb degrade gracefully; eager image
# data is unaffected. Inference must run on GPU (CPU MKL SIGFPEs on this CPU).
# Build with: --build-arg NO_SSE42=1
# NOTE: pyarrow is installed root-owned in /usr/local/lib; this stage runs as the
# unprivileged `biopb` user, so the uninstall must switch to root or it fails
# silently with PermissionError and pyarrow stays in the image.
ARG NO_SSE42=0
USER root
RUN if [ "$NO_SSE42" = "1" ]; then pip uninstall -y pyarrow || true; fi
USER biopb

# Download model checkpoint
RUN wget -q https://huggingface.co/jiyuuchc/ucell/resolve/main/ucell-1024.pt -O model.pt
RUN wget -q https://huggingface.co/jiyuuchc/ucell/resolve/main/ucell-768.pt -O model_s.pt

# Copy service code
COPY --chown=biopb:biopb . .

EXPOSE 50051

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD grpc_health_probe -addr=localhost:50051 || exit 1

ENTRYPOINT ["python", "./ucell_server.py"]
CMD ["--no-token", "--modelpath", "./model.pt"]
