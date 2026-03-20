FROM jiyuuchc/biopb-base:0.2.1

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 \
    libxrender1 \
    libxtst6 \
    && rm -rf /var/lib/apt/lists/*
USER biopb

WORKDIR $HOME

# Download model
RUN wget -q https://huggingface.co/jiyuuchc/lacss3/resolve/main/lacss3-base.pkl -O lacss3-base.pkl

# Create JAX cache directory
RUN mkdir jax_cache

# Copy service code
COPY --chown=biopb:biopb . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install lacss with specific version
ARG VERSION=0.15.2
RUN pip install --no-cache-dir --no-deps lacss==$VERSION

ENV JAX_COMPILATION_CACHE_DIR="./jax_cache"
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPOSE 50051

ENTRYPOINT ["python", "./lacss_server.py"]
CMD ["--no-token", "--f16", "--modelpath", "./lacss3-base.pkl"]