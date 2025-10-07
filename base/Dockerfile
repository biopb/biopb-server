FROM python:3.10-slim-trixie

RUN useradd -m biopb
ENV HOME=/home/biopb

RUN apt-get update && \
    apt-get install -y git

USER biopb
WORKDIR $HOME

ENV PATH="$PATH:$HOME/.local/bin/"

RUN pip install biopb==0.1.3

# copy rest of the code
COPY --chown=biopb:biopb . .


