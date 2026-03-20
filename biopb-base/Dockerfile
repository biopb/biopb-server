# Base image for biopb gRPC services
# Services should inherit from this image: FROM biopb-base:latest

FROM python:3.10-slim-trixie

# Create non-root user and install common system packages
RUN useradd -m biopb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        procps \
        wget \
        && rm -rf /var/lib/apt/lists/*

# Set up environment
ENV HOME=/home/biopb
ENV PATH="$PATH:$HOME/.local/bin"

# Install base Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy common utilities to a location services can import from
RUN mkdir -p /opt/biopb
COPY common.py health.py debug.py server.py test.py test_streaming.py test_image.png /opt/biopb/
ENV PYTHONPATH="/opt/biopb"

# Set working directory
WORKDIR $HOME

# Switch to non-root user
USER biopb

# Expose default port
EXPOSE 50051

# Health check endpoint (can be overridden by services)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:50051/ || exit 1