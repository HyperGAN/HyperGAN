# Python API Reference (`hypergan.api`)

HyperGAN provides an ergonomic Python SDK under `import hypergan as hg`. It features full PEP 561 static typing (`py.typed`), fast torch-free import times (< 15 ms), typed dataclasses, non-blocking execution handles, and programmatic early-stopping.

---

## 1. Quickstart & Import Architecture

```python
import hypergan as hg
```

- **Fast Lazy Loading**: `import hypergan` executes in under 15ms. Heavy dependencies (such as `torch`, `torchvision`, and `onnx`) are lazily loaded on first function call.
- **PEP 561 Compliant**: HyperGAN includes `py.typed` and comprehensive `TYPE_CHECKING` stubs. All public functions and dataclasses are fully resolved by Mypy and Pyright.

---

## 2. Model Manipulation

HyperGAN represents configuration through the immutable `Model` object (wrapping lower-level dictionary recipes):

### `hg.load(path_or_str)`
Loads and lowers a `hypergan-model/1` TOML file:
```python
model = hg.load("examples/cifar10-transgan.toml")
```

### `hg.packaged(name)` & `hg.packaged_list()`
Accesses packaged models bundled directly with the HyperGAN package:
```python
model = hg.packaged("cifar10-transgan")
all_models = hg.packaged_list()
```

### `hg.override(model, overrides)`
Creates a new `Model` with deep-merged dictionary overrides:
```python
tuned = hg.override(model, {
    "training": {"steps": 50000, "batch_size": 64},
    "optimizer": {"lr": 0.0001}
})
```

### `hg.dumps(model)` & `hg.save(model, path)`
Serializes a `Model` to TOML:
```python
toml_str = hg.dumps(model)
hg.save(model, "tuned.toml")
```

---

## 3. Validation & Introspection

### `hg.validate(model) -> list[ValidationWarning]`
Performs structural, shape, and dependency validation, returning a list of `ValidationWarning` objects (which subclass `str` for backward compatibility):
```python
warnings = hg.validate(model)
for w in warnings:
    print(f"Warning: {w.message}")
```

### `hg.schema(section=None, json=False)`
Returns the JSON schema or formatted reference for model configuration tables:
```python
schema = hg.schema("data")
```

### `hg.operators(query=None, json=False)`
Introspects registered HNDL operators:
```python
ops = hg.operators("norm")  # Finds rms_norm, group_norm, layer_norm, etc.
```

---

## 4. Training & Non-Blocking Execution

### `hg.train(model, run_dir=None, steps=None, devices="cuda", on_event=None) -> Run`
Synchronous training runner. Supports direct device routing (`devices="cuda:0"` or `devices="cpu"`) without altering global `CUDA_VISIBLE_DEVICES`:

```python
result = hg.train(
    model,
    run_dir="runs/experiment_1",
    steps=10000,
    devices="cuda:0"
)
print("Trained to step:", result.step)
```

### Programmatic Early Stopping (`on_event`)
The `on_event` callback receives real-time event dictionaries. Return `False` or raise `hg.StopTraining("reason")` to trigger a clean, graceful early stop with checkpoint commit:

```python
def check_divergence(event):
    if event.get("event") == "train":
        metrics = event.get("metrics", {})
        d_loss = metrics.get("loss_d", 0.0)
        if d_loss > 10.0:
            raise hg.StopTraining("Discriminator exploded")
        if event.get("step", 0) >= 5000:
            return False  # Graceful early stop
    return True

result = hg.train(model, on_event=check_divergence)
print("Stop reason:", result.stop_reason)
```

### Non-Blocking Launch (`hg.launch`)
Launches training asynchronously in a dedicated worker thread, returning a `RunHandle`:

```python
handle = hg.launch(model, run_dir="runs/async_run", steps=20000)

# Check execution status (both is_running() and is_alive() are supported)
print("Running?", handle.is_running())

# Wait with timeout or join (both join() and wait() are supported)
result = handle.join(timeout=300.0)
```

---

## 5. Run Telemetry & Management

### `hg.read_run_info(run_dir) -> RunInfo`
Reads durable metadata and metrics from `manifest.json`, `summary.json`, and `tag.json`:

```python
info = hg.read_run_info("runs/experiment_1")
print(f"Run ID: {info.run_id}, Status: {info.status}, Steps: {info.step}")
print("Stop reason:", info.stop_reason)
```

### `hg.ps(all_runs=False) -> PsSnapshot`
Lists active and past runs across the workspace (returns `PsSnapshot` supporting both mapping and attribute access on `runs`):

```python
ps = hg.ps()
for run in ps.runs:
    print(f"[{run.slug}] Step {run.step} - {run.status}")
```

### `hg.logs(target, tail=50)` & `hg.tag(target, name)`
Inspects output logs or tags runs:
```python
lines = hg.logs("runs/experiment_1", tail=100)
hg.tag("runs/experiment_1", "baseline_v1")
```

---

## 6. Sampling & Deployment

### `hg.sample(...) -> Sample`
Generates samples from a checkpoint or run directory:
```python
sample = hg.sample(
    "runs/experiment_1",
    count=16,
    seed=42,
    output="samples/grid.png",
    device="cuda"
)
print("Saved sample to:", sample.path)
```

### `hg.export(...)`
Exports a trained run to a standalone format:
```python
hg.export("runs/experiment_1", output="models/gen.pt", format="bundle")
hg.export("runs/experiment_1", output="models/gen.safetensors", format="safetensors")
hg.export("runs/experiment_1", output="models/gen.onnx", format="onnx")
```

### `hg.load_bundle(...) -> LoadedGenerator`
Loads an exported `.pt` bundle as an autonomous `torch.nn.Module`:
```python
generator = hg.load_bundle("models/gen.pt", device="cuda")
images = generator.sample(count=8, seed=42)
walk = generator.interpolate(seed_a=100, seed_b=200, steps=10, mode="slerp")
```

---

## 7. Typed Dataclass Reference

### `Run`
```python
@dataclass(frozen=True)
class Run:
    path: Path
    manifest: dict
    status: Optional[str]
    step: Optional[int]
    stop_reason: Optional[str]
    config: Optional[dict]
    fingerprint: Optional[str]
    run_id: str
    short_id: str
    name: str
    slug: str
```

### `RunHandle`
```python
class RunHandle:
    run_dir: Path
    run: Run
    run_info: RunInfo

    def is_alive(self) -> bool: ...
    def is_running(self) -> bool: ...  # Alias for is_alive
    def wait(self, timeout: Optional[float] = None) -> Run: ...
    def join(self, timeout: Optional[float] = None) -> Run: ...  # Alias for wait
    def stop(self, timeout: float = 15.0) -> None: ...
```

### `ValidationWarning`
```python
@dataclass
class ValidationWarning(str):
    message: str
    field: Optional[str] = None
    level: str = "warning"
```
*(Subclasses `str` for backward compatibility with string-based assertions and formatting).*

### `RunStatus`
```python
@dataclass
class RunStatus(dict):
    run_id: str
    status: str              # "running", "completed", "stopped", "failed"
    step: Optional[int] = None
    total_steps: Optional[int] = None
    percent: Optional[float] = None
    steps_per_sec: Optional[float] = None
    eta_seconds: Optional[float] = None
    devices: str = "cpu"
    best_metric: Optional[dict[str, Any]] = None
    recent_checkpoints: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    slug: Optional[str] = None
```
*(Subclasses `dict` with `__getattr__` attribute access for `slug`, `step`, `status`, `device`, `pid`, etc.).*

### `PsSnapshot`
```python
@dataclass
class PsSnapshot(dict):
    runs: list[RunStatus]
    queued: list[dict[str, Any]]
    active_count: int
    total_known: int
```
*(Implements dictionary mapping protocol, `__len__`, and property aliases `total_count` and `completed_count`).*

### `TrainResult`
```python
@dataclass
class TrainResult:
    run: Any
    step: int = 0
    status: str = "completed"
    metrics: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[Path] = None
    stop_reason: Optional[str] = None
    run_dir: str             # Property
    run_id: str              # Property
    summary: dict[str, Any]  # Property alias for metrics
```

### `Sample` & `SampleResult`
```python
@dataclass(frozen=True)
class Sample:
    step: int
    name: str
    data: Optional[Any]      # torch.Tensor
    inputs: dict[str, Any]
    path: Path
    png: Optional[bytes] = None

@dataclass
class SampleResult:
    path: Path
    count: int = 0
    step: int = 0
    png: bytes | None = None
    data: Optional[Any] = None
    inputs: dict[str, Any] = field(default_factory=dict)
```
