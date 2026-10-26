# Tests

Run everything from the repository root after `python -m pip install -e '.[dev,train,image,cifar,web]'` (plus a CPU or CUDA PyTorch wheel; see [CONTRIBUTING.md](../CONTRIBUTING.md)).

## Layout

| Directory | What it covers | Needs |
| --- | --- | --- |
| `foundation/` | Lightweight package: CLI parsing, model files, registry, queue, packaging and docs consistency. Must pass without PyTorch. | core install |
| `reference/` | Numerical reference on CPU: training steps, bit-exact stop/resume, checkpoints, metrics, replicated CPU workers. | `train`, `image`, `cifar` |
| `unit/` | Focused module tests: aggregation strategies, profiles, multi-node topology, qualify, RunPod runner. | `train` |
| `playbooks/` | User journeys through the CLI and Python API (new, validate, train, inspect, sample, export, rm). | `train`, `web` |
| `e2e/` | Multi-device acceptance suite (Issue #186): replicated profiles, aggregation, multi-node TCP, qualification. | `train` |
| `e2e_integration/` | Foldlite DB reducer integration. | `reducers`, `web` |
| `web/` | Viewer HTTP API and local service. | `web`, `httpx` |
| `browser/` | Headless Chromium checks of the viewer. | `web`, `playwright` |
| `reducers/` | Shared Foldlite reducers. | `reducers` |
| `cuda/` | Native CUDA and two-GPU NCCL tests. Never skip missing hardware: they fail without GPUs. | CUDA runtime, GPUs |

Shared helpers sit beside the directories: `hndl_fixtures.py` and `dinov3_fixtures.py` (network fixtures), `distributed_fixtures.py` (toy recipe and the FedAvg/DiLoCo oracles), `config_fixtures.py` (engine-recipe TOML writers), and `fixtures/` (TOML inputs). `e2e/harness.py` holds the e2e-only builders and the multi-rank smoke runner.

## Markers

- `heavy`: the test starts real operating system processes (a CLI run, a worker service, a multi-rank job) and takes a second or more. The default run deselects these and reports them as deselected.
- `multi_gpu`: `tests/cuda` tests that need two or more visible GPUs.

Tests skip only for missing hardware (for example the two-GPU e2e cases) or a missing optional extra they name. A missing feature or a broken import is a failure, never a skip.

## Running each tier

```sh
python -m pytest                       # fast suite: the directories in pyproject.toml testpaths, without heavy
python -m pytest -m heavy              # only the heavy tests from the same directories
python -m pytest -m ""                 # both halves
python -m pytest tests/web tests/reducers tests/e2e_integration
python -m pytest tests/browser         # after: python -m playwright install chromium
python -m pytest tests/cuda -m ""      # both fast and heavy CUDA tests; two NVIDIA GPUs; see docs/cuda.md
```

The default `testpaths` are `foundation`, `reference`, `unit`, `playbooks`, `e2e` and `e2e_integration`. The `web`, `browser`, `reducers` and `cuda` directories run only when you name them.

CI runs these against the built wheel from outside the checkout (`python -I -m pytest <abs path> --import-mode=importlib`): the lightweight job runs `foundation` on Linux, macOS and Windows for Python 3.11–3.14; the CPU reference job runs `reference`, `unit`, `playbooks` and `e2e`; the heavy jobs run the `heavy` half for `master` pull requests and releases. See [.github/workflows/ci.yml](../.github/workflows/ci.yml).

When you add a test that starts subprocesses, time it with `--durations=0` and mark it `heavy` if it takes a second or more.
