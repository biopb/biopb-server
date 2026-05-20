FROM jiyuuchc/biopb-image-base:latest

WORKDIR $HOME

# Install Python dependencies
COPY --chown=biopb:biopb requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pull model data
RUN python -c 'from cellpose import models; models.CellposeModel(gpu=True)'

# Copy service code
COPY --chown=biopb:biopb . .

EXPOSE 50051

# Health check using grpc_health_probe (available in base image)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD grpc_health_probe -addr=localhost:50051 || exit 1

ENTRYPOINT ["python", "./cellpose_server.py"]
CMD ["--no-token"]
