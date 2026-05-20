FROM jiyuuchc/biopb-image-base:latest

USER biopb

WORKDIR $HOME

COPY --chown=biopb:biopb requirements.txt ./
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt

RUN pip install --no-cache-dir --ignore-requires-python git+https://github.com/jiyuuchc/ucell.git

# Download model checkpoint
RUN wget -q https://huggingface.co/jiyuuchc/ucell/resolve/main/ucell-1024.pt -O model.pt
RUN wget -q https://huggingface.co/jiyuuchc/ucell/resolve/main/ucell-768.pt -O model_s.pt

# Copy service code
COPY --chown=biopb:biopb . .

EXPOSE 50051

# Health check using grpc_health_probe (available in base image)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD grpc_health_probe -addr=localhost:50051 || exit 1

ENTRYPOINT ["python", "./ucell_server.py"]
CMD ["--no-token", "--modelpath", "./model.pt"]
