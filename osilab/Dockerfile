FROM jiyuuchc/biopb-base:0.2.1

WORKDIR $HOME

# Install Python dependencies
COPY --chown=biopb:biopb requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code (includes model files)
COPY --chown=biopb:biopb . .

EXPOSE 50051

ENTRYPOINT ["python", "./segformer_server.py"]
CMD ["--no-token"]