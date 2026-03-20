FROM jiyuuchc/biopb-base:0.2.1

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

ENTRYPOINT ["python", "./samcell_server.py"]
CMD ["samcell-generalist.pt", "--no-token"]