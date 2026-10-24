# Container Profile (`Dockerfile`)

HyperGAN provides a pinned, multi-stage OCI container profile in [`Dockerfile`](../Dockerfile) built from the installed wheel rather than a raw repository checkout.

## Container Contract

- **Base image:** `python:3.12-slim-bookworm` (multi-stage `builder` and `runtime` stages)
- **Installed artifact:** The `builder` stage builds a wheel into `/dist` (`python -m pip wheel --no-cache-dir --no-deps --wheel-dir /dist .`), and the `runtime` stage installs that wheel into the system environment.
- **Unprivileged user:** Runs as `hypergan:hypergan` (`uid=1000`, `gid=1000`) with working directory `/workspace`.
- **Mount points:**
  - `/data`: Input dataset directory (mount read-only when training or validating image folders).
  - `/runs`: Output run directory and persistent global registry (`HYPERGAN_STATE_DIR=/runs/.hypergan`).
  - `/workspace`: Project working directory for `.toml` recipes and `hypergan.toml` workspace configs.
- **Entrypoint & signals:**
  - `ENTRYPOINT ["python", "-m", "hypergan"]` (exec form, so `python -m hypergan` runs as PID 1 without an intermediate shell wrapper and preserves exit codes).
  - `CMD ["--help"]`
  - `STOPSIGNAL SIGINT`, matching `hypergan.run_signals.GracefulStop` so `docker stop` triggers a clean checkpoint-and-exit at the next update boundary.

## Building the Image

### 1. Lightweight CLI Image (Default)

The default build installs the zero-dependency core package (`new`, `validate`, `inspect`, `ps`, `status`, `logs`, `runs`, `diff`, `report`, `doctor`):

```sh
docker build -t hypergan:cli .
```

Verify the image outside the source tree:

```sh
docker run --rm hypergan:cli --version
docker run --rm hypergan:cli --help
```

### 2. Training & Web Observatory Image

Pass `--build-arg EXTRAS=...` to install optional runtime extras (for example `train,image,web`):

```sh
docker build --build-arg EXTRAS=train,image,web -t hypergan:train .
```

## Data, Workspace, and Run Mounts

Mount `/data` read-only and `/runs` read-write so training artifacts, checkpoints, and the run registry (`HYPERGAN_STATE_DIR=/runs/.hypergan`) persist on the host:

```sh
mkdir -p ./workspace ./runs

# Scaffold a starter project inside /workspace sized from /data
docker run --rm \
  --mount type=bind,src="$(pwd)/data",dst=/data,readonly \
  --mount type=bind,src="$(pwd)/runs",dst=/runs \
  --mount type=bind,src="$(pwd)/workspace",dst=/workspace \
  hypergan:cli new --project /workspace/my_gan --dataset /data

# Validate the generated configuration and dataset inventory
docker run --rm \
  --mount type=bind,src="$(pwd)/data",dst=/data,readonly \
  --mount type=bind,src="$(pwd)/runs",dst=/runs \
  --mount type=bind,src="$(pwd)/workspace",dst=/workspace \
  hypergan:train validate /workspace/my_gan/configs/dcgan.toml --data
```

## Running Training, Viewer, and Signals

### Headless Training (`--no-server`)

For batch or automated container runs without a web listener, pass `--no-server`:

```sh
docker run --rm \
  --mount type=bind,src="$(pwd)/data",dst=/data,readonly \
  --mount type=bind,src="$(pwd)/runs",dst=/runs \
  --mount type=bind,src="$(pwd)/workspace",dst=/workspace \
  hypergan:train train /workspace/my_gan/configs/dcgan.toml \
  --run-dir /runs/dcgan-exp1 \
  --steps 1000 \
  --no-server
```

### Publishing the Web Observatory (`-p 8765:8765`)

The container exposes port `8765`. Publish `-p 8765:8765` to access the Web Observatory from the host browser:

```sh
docker run --rm \
  -p 8765:8765 \
  --mount type=bind,src="$(pwd)/runs",dst=/runs \
  hypergan:train view /runs --host 0.0.0.0 --port 8765 --no-browser
```

### Graceful Stop (`STOPSIGNAL SIGINT`)

Because `Dockerfile` sets `STOPSIGNAL SIGINT` and uses the exec-form `ENTRYPOINT ["python", "-m", "hypergan"]`, stopping a running training container with `docker stop` (or `docker kill --signal=SIGINT`) delivers `SIGINT` directly to PID 1:

```sh
docker stop --time 30 <container_id>
```

HyperGAN catches `SIGINT`, finishes the current update, writes an atomic checkpoint and updated `manifest.json` (`status = "stopped"`), and exits cleanly before Docker's grace period expires.

### GPU Pass-Through (`--gpus all`)

When running with a CUDA-enabled PyTorch installation and the NVIDIA Container Toolkit on Linux, pass `--gpus all`:

```sh
docker run --rm --gpus all \
  --mount type=bind,src="$(pwd)/data",dst=/data,readonly \
  --mount type=bind,src="$(pwd)/runs",dst=/runs \
  --mount type=bind,src="$(pwd)/workspace",dst=/workspace \
  hypergan:train doctor
```

## Qualification Boundaries

- The default `Dockerfile` validates the installed package, CLI entrypoints, volume mounts, and signal handling on `python:3.12-slim-bookworm`.
- Multi-GPU and multi-node cluster container profiles follow their respective hardware and cluster qualification gates in [`docs/distributed.md`](distributed.md) and [`docs/cuda.md`](cuda.md).
