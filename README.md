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


Every image serves the biopb.image `Ops` protocol (`--help` lists the flags; `--describe`
prints the ops an image offers and exits).

Locally, share the host's network and bind loopback, where no token is needed:

``` sh
docker run --gpus=all --network host <image-name> --host 127.0.0.1 --port 50051
```

From the network, publish the port. The image binds `0.0.0.0` by default, where a token
is required: set `$BIOPB_ALGORITHM_TOKEN`, or the server mints one and prints it at
startup. Clients send it as `authorization: Bearer <token>`.

``` sh
docker run --gpus=all -p 50051:50051 -e BIOPB_ALGORITHM_TOKEN=<token> <image-name>
```

Debug logging: add `-e BIOPB_LOG_LEVEL=DEBUG`.

Results too large for one message are returned through an embedded tensor server when
the container is started with `--cache-dir <dir>` (and `--tensor-port`, default 8817,
published).

Note: Default transport is HTTP (no encryption). To use TLS, setup a reverse proxy server, e.g., Nginx, to forward gRPC calls.

## License

This repository is licensed under the MIT License (see [LICENSE](LICENSE)), with
one exception: the `unifmir/` service vendors model code from the GPL-3.0-licensed
[UNiFMIR](https://github.com/cxm12/UNiFMIR) project, so the `unifmir/` directory is
distributed under the GNU GPL v3.0. See [unifmir/LICENSE](unifmir/LICENSE) and
[unifmir/NOTICE](unifmir/NOTICE) for details.

