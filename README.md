## Docker Images

The folder contains code for building the docker images of biopb algorithm servers.

### Pre-built public images
  - jiyuuchc/cellpose: [Cellpose Cyto3](https://cellpose.com)
  - jiyuuchc/cellpose-sam: [Cellpose-SAM](https://cellpose.com)
  - jiyuuchc/ucell: [ucell](https://github.com/jiyuuchc/ucell)
  - jiyuuchc/unifmir: [UNiFMIR](https://github.com/cxm12/UNiFMIR)

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

## License

This repository is licensed under the MIT License (see [LICENSE](LICENSE)), with
one exception: the `unifmr/` service vendors model code from the GPL-3.0-licensed
[UNiFMIR](https://github.com/cxm12/UNiFMIR) project, so the `unifmr/` directory is
distributed under the GNU GPL v3.0. See [unifmr/LICENSE](unifmr/LICENSE) and
[unifmr/NOTICE](unifmr/NOTICE) for details.

