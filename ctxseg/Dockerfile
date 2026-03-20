FROM jiyuuchc/biopb-base:0.2.1

WORKDIR $HOME

# Install Python dependencies
COPY --chown=biopb:biopb requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY --chown=biopb:biopb . .

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPOSE 50051

ENTRYPOINT ["python", "./ctxseg_server.py"]
CMD ["--no-token", "ctxseg_p.pickle"]