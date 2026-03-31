FROM jiyuuchc/biopb-base:0.2.4

WORKDIR $HOME

# Install Python dependencies
COPY --chown=biopb:biopb requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pull model data
RUN python -c 'from cellpose import models; models.Cellpose("cyto3")'

# Copy service code
COPY --chown=biopb:biopb . .

EXPOSE 50051

ENTRYPOINT ["python", "./cellpose_server.py"]
CMD ["--no-token"]
