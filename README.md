## Docker Images

The folder contains code for building the docker images of biopb servers.

### Pre-built public images
  - jiyuuchc/cellpose: [Cellpose Cyto3](https://cellpose.com)
  - jiyuuchc/cellpose-sam: [Cellpose-SAM](https://cellpose.com)
  - jiyuuchc/lacss: [Lacss3-base](https://github.com/jiyuuchc/lacss)
  - jiyuuchc/samcell: [Finetuned SAM model](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0319532)

### Run

Requirements:
  - NVIDIA kernel driver (>=525)
  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)


``` sh
docker run --gpus=all -p 50051:50051 <image-name>

# running on a different port
docker run --gpus=all -p 8888:50051 <image-name>

# only accepting local RPC call
docker run --gpus=all -p 50051:50051 <image-name> --local

# debug mode
docker run --gpus=all -p 50051:50051 <image-name> --no-token --debug

# require a token to access
docker run --gpus=all -p 50051:50051 <image-name> --token
```

Note: Default transport is HTTP (no encryption). To use HTTPS, run container in "--local" mode and setup a reverse proxy server, e.g., Nginx, to forward RPC calls.

