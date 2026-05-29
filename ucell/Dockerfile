FROM jiyuuchc/biopb-image-base:e7aa1fc

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
