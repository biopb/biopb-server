FROM jiyuuchc/biopb-image-base:sha-d38e184

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

# Off loopback the server requires a token: $BIOPB_ALGORITHM_TOKEN, or one it
# mints and prints. Bound to 127.0.0.1 (with --network host) it takes none.
ENTRYPOINT ["python", "./ucell_server.py"]
CMD ["--host", "0.0.0.0"]
