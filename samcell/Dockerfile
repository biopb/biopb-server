FROM jiyuuchc/biopb-image-base:latest

WORKDIR $HOME

# Install Python dependencies
COPY --chown=biopb:biopb requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY --chown=biopb:biopb . .

# Download model and initialize
RUN python -c "from model import FinetunedSAM; _ = FinetunedSAM('facebook/sam-vit-base', finetune_vision=False, finetune_prompt=True, finetune_decoder=True);"
RUN wget -q https://github.com/saahilsanganeriya/SAMCell/releases/download/v1/samcell-generalist.pt

EXPOSE 50051

# Health check using grpc_health_probe (available in base image)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD grpc_health_probe -addr=localhost:50051 || exit 1

ENTRYPOINT ["python", "./samcell_server.py"]
CMD ["samcell-generalist.pt", "--no-token"]