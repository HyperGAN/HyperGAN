# Google Colab benchmark orchestrator

`scripts/colab_bench.py` runs HyperGAN training benchmarks and standalone EMA bundle evaluations across up to three warm Google Colab GPU VMs (`T4`, `L4`, `G4`, `A100`, `H100`) and pulls complete run directories back into `runs/<run_name>/` so every Colab run appears immediately in the local [Web Observatory](local-web.md).

## Safety guarantees

- **Hard 3-GPU concurrency cap**: `--max-workers` defaults to `3` and refuses values greater than `3` unless `--allow-over-cap` is passed explicitly. Single-job runs always allocate only `1` VM.
- **Isolated per-worker session configs + staggered launches**: every worker writes its own `--config <worker>.colab.json` under `runs/colab/` and staggers `colab new` calls by `--stagger-seconds 20` to avoid `sessions.json.lock` contention and Colab 503 allocation bursts.
- **Two-layer crash-proof teardown**:
  1. Local `try/finally`, `atexit`, `SIGINT`, and `SIGTERM` handlers stop every provisioned VM immediately and persist session metadata under `runs/colab/sessions/`.
  2. Immediately after boot, a detached VM-side `/content/watchdog.py` is armed (with 3 retries; failure terminates the VM) and sends `POST http://$TBE_RUNTIME_ADDR/unassign` if the local controller disconnects after job completion (`--idle-grace-minutes`), training logs/events stall (`--stall-minutes`), or total wall time exceeds `--max-minutes`.
- **OS process-group hard timeouts**: every `colab` CLI subprocess runs in its own session (`start_new_session=True`) with guaranteed `SIGTERM -> SIGKILL` process-group cleanup so hung websocket threads never leak idle GPU credits.
- **Cost & hardware telemetry**: records requested and actual assigned CUDA model, wall seconds, GPU-hours, and estimated Compute Units (CU) per worker and per batch in `runs/colab/<batch>/summary.json`. Pass `--fallback-gpus A100,L4,T4` (or `L4,T4`) to automatically fall back through an ordered accelerator list if the primary pool returns transient `503 Service Unavailable` responses.

| Accelerator | Typical VRAM | Est. CU / hr |
| --- | --- | --- |
| `cpu` | — | 0.07 |
| `T4` | 15.6 GB | 1.15 |
| `L4` | 23.7 GB | 1.94 |
| `G4` | 24.0 GB | 3.00 |
| `A100` | 40.0 GB | 6.80 |
| `H100` | 80.0 GB | 12.00 |

## Prerequisites

Authenticate the `google-colab-cli` binary (`colab`) locally and verify zero straggler sessions before launching benchmarks:

```sh
python scripts/colab_bench.py status
```

## Running training benchmarks

### Single configuration

Run one HyperGAN TOML configuration on a Colab L4 GPU and pull all run artifacts (`summary.json`, `manifest.json`, `model.json`, `events.jsonl`, `metrics/`, `previews/`, EMA `model.pt`, `train.log`, and the latest checkpoint) into `runs/<run_name>/`:

```sh
python scripts/colab_bench.py run \
  --config cifar10-tiny-transformer \
  --run-name cifar10-smoke-l4 \
  --gpu L4 \
  --workers 1 \
  --stop-after-steps 50 \
  --checkpoint-every 50 \
  --preview-every 50
```

### Multi-job JSON or TOML manifest

Queue multiple experiments across up to 3 warm Colab GPU VMs. Each warm VM bootstraps HyperGAN once (`particlegan==0.12.0`, `hndl`, `torch-fidelity==0.4.0`, `foldlitedb`, plus editable HyperGAN from `--ref` or a local bundle/tarball), automatically caches CIFAR-10 (`hypergan download-data cifar10`) or 64×64 benchmark datasets (`python scripts/prepare_benchmark_dataset.py --dataset ffhq64` / `flowers102`) once per VM, and drains all matching-GPU jobs from the queue:

```toml
[defaults]
gpu = "A100"
stop_after_steps = 50000
checkpoint_every = 5000
chunk_steps = 10000
max_retries = 2

[[jobs]]
run_name = "cifar10-resnet-v12"
config = "benchmarks/whitepaper/cifar10-resnet-scratch-v12.toml"

[[jobs]]
run_name = "cifar10-resnet-adam"
config = "benchmarks/whitepaper/cifar10-resnet-scratch-adam.toml"
```

Preview the schedule without booting any VM using `--dry-run`, then launch:

```sh
python scripts/colab_bench.py run --manifest benchmarks/queue.toml --workers 3 --dry-run
python scripts/colab_bench.py run --manifest benchmarks/queue.toml --workers 3 --ref develop
```

## Disconnect resilience, resume, and collapse telemetry

Long 50k–100k step runs can survive Colab preemptions and runtime resets via three complementary mechanisms:

- **Chunked local sync (`--chunk-steps N`)**: runs `hypergan train` / `hypergan resume` in `N`-step chunks and pulls the updated run directory and latest checkpoint back to `runs/<run_name>/` after every chunk.
- **Background periodic checkpoint sync (`--sync-interval-minutes M`)**: pulls the latest checkpoint and summary back locally every `M` minutes during continuous single-attempt runs while refreshing the VM watchdog heartbeat.
- **Automatic preemption recovery (`--max-retries K`) + optional Google Drive mirror (`--drive-dir`)**: if a Colab VM disconnects mid-run, the orchestrator tears down the broken session, provisions a fresh VM, uploads the last synced local checkpoint, and continues via `hypergan resume`.

Completed runs whose local `runs/<run_name>/summary.json` already reports `status == "stopped"` or `"complete"` at or beyond the target step are skipped automatically unless `--force` is passed. If a baseline diverges or produces NaNs, all partial logs, previews, and metrics are still pulled back and flagged (`nan_detected`, `collapse_detected`) in the batch summary.

Open pulled-back runs directly in the local Observatory:

```sh
hypergan view runs/
```

## Evaluating exported EMA bundles

Upload one or more local EMA `model.pt` bundles (or run directories containing `model.pt`) to a single Colab GPU VM, run pinned publication evaluators via `python -m hypergan.image_metrics` (`InceptionSuite`, `DINOv2Suite`, `InceptionFID`, `DINOv2FD`) or custom `module:Class` snapshot factories, download structured JSON receipts, and optionally import them back into each local run directory:

```sh
python scripts/colab_bench.py eval-bundles \
  --bundle runs/cifar10-resnet-v12/model.pt \
  --bundle runs/cifar10-resnet-adam/model.pt \
  --metrics InceptionSuite,DINOv2Suite \
  --dataset cifar10 \
  --samples 50000 \
  --gpu L4 \
  --import-to-run
```

## Inspecting status and cleaning up stragglers

Inspect tracked session records under `runs/colab/sessions/`, active Colab runtimes, and Compute Unit balance:

```sh
python scripts/colab_bench.py status
```

Tear down any active or orphaned Colab VMs immediately:

```sh
python scripts/colab_bench.py cleanup --dry-run
python scripts/colab_bench.py cleanup
```
