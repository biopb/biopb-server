FROM jiyuuchc/biopb-base:0.1.4
USER root
RUN apt-get install -y wget

ENV HOME=/home/biopb
USER biopb
WORKDIR $HOME

COPY --chown=bipob:biopb requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy rest of the code
COPY --chown=biopb:biopb . .

ENV PYTHONPATH=.
RUN python -c "from model import FinetunedSAM; _ = FinetunedSAM('facebook/sam-vit-base', finetune_vision=False, finetune_prompt=True, finetune_decoder=True);"

RUN wget https://github.com/saahilsanganeriya/SAMCell/releases/download/v1/samcell-generalist.pt

# fix file permission 
RUN chmod -R 777 $HOME

EXPOSE 50051

ENTRYPOINT ["python", "./server.py"]

CMD [ "samcell-generalist.pt", "--no-token" ]
