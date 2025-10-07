FROM jiyuuchc/biopb-base:0.1.3

ENV HOME=/home/biopb
USER biopb
WORKDIR $HOME

COPY --chown=bipob:biopb requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy rest of the code
COPY --chown=biopb:biopb . .

# fix file permission 
RUN chmod -R 777 $HOME

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPOSE 50051

ENTRYPOINT ["python", "./server.py"]

CMD [ "--no-token", "model_d.pickle" ]
