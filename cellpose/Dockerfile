FROM jiyuuchc/biopb-image-base:sha-d38e184

WORKDIR $HOME

# Install Python dependencies
COPY --chown=biopb:biopb requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pull model data
RUN python -c 'from cellpose import models; models.Cellpose("cyto3")'

# Copy service code
COPY --chown=biopb:biopb . .

EXPOSE 50051

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD grpc_health_probe -addr=localhost:50051 || exit 1

# Off loopback the server requires a token: $BIOPB_ALGORITHM_TOKEN, or one it
# mints and prints. Bound to 127.0.0.1 (with --network host) it takes none.
ENTRYPOINT ["python", "./cellpose_server.py"]
CMD ["--host", "0.0.0.0"]
