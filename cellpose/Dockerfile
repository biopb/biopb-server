FROM jiyuuchc/biopb-image-base:ec58ca5

WORKDIR $HOME

# Install Python dependencies
COPY --chown=biopb:biopb requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Optional: build for CPUs without SSE4.2/AVX (e.g. old AMD Opterons).
# pyarrow's wheels are compiled with an SSE4.2 baseline and SIGILL on import on
# such CPUs. It is only needed for the lazy/Flight (TensorFlightClient) data
# path, which those CPUs cannot serve anyway. Removing it lets pandas/dask and
# biopb fall back gracefully (biopb imports pyarrow lazily). Eager/pixels image
# data is unaffected. Build with: --build-arg NO_SSE42=1
# NOTE: pyarrow is installed root-owned in /usr/local/lib; the base image runs as
# the unprivileged `biopb` user, so the uninstall must switch to root or it fails
# silently with PermissionError and pyarrow stays in the image.
ARG NO_SSE42=0
USER root
RUN if [ "$NO_SSE42" = "1" ]; then pip uninstall -y pyarrow || true; fi
USER biopb

# Pull model data
RUN python -c 'from cellpose import models; models.Cellpose("cyto3")'

# Copy service code
COPY --chown=biopb:biopb . .

EXPOSE 50051

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD grpc_health_probe -addr=localhost:50051 || exit 1

ENTRYPOINT ["python", "./cellpose_server.py"]
CMD ["--no-token"]
