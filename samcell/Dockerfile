FROM jiyuuchc/biopb-base:0.1.4

ENV HOME=/home/biopb
USER biopb
WORKDIR $HOME

COPY --chown=bipob:biopb requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy rest of the code
COPY --chown=biopb:biopb . .

# fix file permission 
RUN chmod -R 777 $HOME

ENV PYTHONPATH=.
RUN python -c "from model import FinetunedSAM; _ = FinetunedSAM('facebook/sam-vit-large', finetune_vision=False, finetune_prompt=True, finetune_decoder=True);"

EXPOSE 50051

ENTRYPOINT ["PYTHONPATH=.", "python", "./server.py"]

CMD [ "samcell-generalist-large.pt", "--no-token" ]
