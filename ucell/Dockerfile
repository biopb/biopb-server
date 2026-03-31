FROM jiyuuchc/biopb-base:0.2.4

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

ENTRYPOINT ["python", "./ucell_server.py"]
CMD ["--no-token", "--modelpath", "./model.pt"]
