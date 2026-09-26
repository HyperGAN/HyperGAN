# Python API proposal — 2026-09-26

Status: **proposal for owner review.** Nothing in this document is implemented.
It sets out a public Python API for HyperGAN 2, grounded in the code on
`develop` at `370f6306`, and ends with the decisions the owner needs to make
before implementation starts.

## Why someone would use HyperGAN instead of ParticleGAN directly

ParticleGAN supplies the formulation: the particle/MoG prior, K3P critic
penalty, VICReg and the recipe optimizers. Everything around that formulation
is what HyperGAN adds. The API should make those additions the obvious reason
to `pip install hypergan`:

1. **Recipes that work.** A named, measured starting point: data in, images
   out, with no need to choose losses, penalties, learning rates or schedules.
2. **Your own networks, with the rest supplied.** Swap in a generator or
   discriminator (an `nn.Module` or HNDL source). HyperGAN keeps the prior,
   losses, optimizers, EMA, schedule, checkpoints, previews and metrics.
3. **Access to training as it runs.** Metrics, sample grids, evaluations
   such as FID, and the live viewer, from Python, while the run is going or
   after it finishes.
4. **Runs that survive.** Complete checkpoints, exact resume and
   preemption-safe stops. Training on spot instances or a shared workstation
   just works.
5. **More GPUs from one argument.** `gpus=2` rather than a different program.
6. **A generator you can use.** Load a trained model as an `nn.Module`, sample
   it, move through its latent space, and export it.

Items 4–6 answer "what else": they already exist on `develop` (4 fully, 5 for
the toy recipe, 6 partly) but only behind the CLI.

## Design principles

- **A recipe is data.** Everything the API builds lowers to the same resolved
  configuration dictionary the CLI builds from TOML. Fingerprints, exact
  resume, replicated execution, the viewer and the run manifest therefore work
  unchanged, and `recipe.save("config.toml")` gives the CLI the same run.
  Live Python objects are never baked into a run. A network is recorded by
  import path and constructor arguments, as `factory = "module:Object"`
  already is.
- **The run directory is the interface.** A `Run` object reads the directory
  and does not own the trainer. The same object works in the training process,
  in a notebook watching a background run, or on another machine reading a
  shared disk.
- **One execution path.** The API calls the code the CLI calls today
  (`execution.prepare_train`, `artifacts.sample`, `metric_evaluation.evaluate`,
  `run_events`). The CLI should later be rewritten as a thin layer over this
  API, not the other way round.
- **Importing stays light.** `import hypergan` imports no torch. Only building,
  training and sampling do, as today.
- **Honest capability.** When a request is outside what has been qualified,
  the API fails early and says so. Examples are an unsupported recipe on
  `gpus=2`, a network defined in `__main__` for a background run, or data
  that cannot resume. The existing qualification warnings carry through.

## The API by example

### 1. Train a recipe on a folder of images

```python
import hypergan as hg

run = hg.train("~/data/faces", recipe="image-64", run_dir="runs/faces")
run.sample(16, seed=1).save("faces.png")
```

`hg.train` blocks until the run finishes (or stops at a budget) and returns a
`Run`. It starts the viewer as `hypergan train` does. The URL is printed and
is also `run.url`. The folder becomes the recipe's `image_folder` data at the
recipe's resolution. The CLI equivalent restores 1.x's one-liner:

```sh
hypergan train ~/data/faces --recipe image-64 --run-dir runs/faces
```

A run directory that already exists means resume, as it does today. Calling
the same line again continues the run.

### 2. Change a few settings

```python
recipe = hg.recipe("image-64", steps=100_000, batch_size=32)
recipe = recipe.replace({"optimizer.lr": 1e-4, "prior.args.num_particles": 8192})
print(recipe)            # resolved TOML, including every default
recipe.save("faces.toml")  # the CLI trains exactly this
run = hg.train("~/data/faces", recipe=recipe, run_dir="runs/faces-lr1e-4")
```

A small set of common settings are keyword arguments: `steps`, `batch_size`,
`lr`, `z_dim`, `seed` and `device`. Everything else uses dotted keys that
mirror the TOML sections. Unknown keys fail, as unknown TOML fields already
do. A `Recipe` is immutable; `replace` returns a new one.

### 3. Train your own network

```python
# mynets.py
from torch import nn

class Generator(nn.Module):
    def __init__(self, z_dim=128, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, width * 8 * 4 * 4), nn.Unflatten(1, (width * 8, 4, 4)),
            nn.ReLU(), nn.ConvTranspose2d(width * 8, width * 4, 4, 2, 1),   # 8
            nn.ReLU(), nn.ConvTranspose2d(width * 4, width * 2, 4, 2, 1),   # 16
            nn.ReLU(), nn.ConvTranspose2d(width * 2, width, 4, 2, 1),       # 32
            nn.ReLU(), nn.ConvTranspose2d(width, 3, 4, 2, 1), nn.Tanh())    # 64

    def forward(self, z):
        return self.net(z)
```

```python
# train.py
import hypergan as hg
from mynets import Generator

recipe = hg.recipe("image-64").replace(generator=hg.Network(Generator, width=96))
recipe.check()   # builds on the meta device and checks every shape; no GPU, no data
run = hg.train("~/data/faces", recipe=recipe, run_dir="runs/my-generator")
```

`hg.Network(cls, **kwargs)` records `factory = "mynets:Generator"` and
`args = {width = 96}`. The recipe owns the input/output contract:

| Role | Receives | Must return |
| --- | --- | --- |
| generator | `z`: `[B, z_dim]` from the prior | image `[B, C, H, W]` in `[-1, 1]` |
| discriminator | `x`: `[B, C, H, W]` | score `[B, 1]`, unbounded logits |

`recipe.check()` names any mismatch before a GPU or dataset is touched. A
discriminator in training mode must not use BatchNorm, because the critic
penalty differentiates each example's score. `check()` reports this rule
instead of leaving it to a paper.

HNDL source works the same way. The recipe fills in the shapes, so only the
layers are written:

```python
recipe = hg.recipe("image-64").replace(discriminator=hg.HNDL("""
conv(64, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
conv(128, kernel_size=4, stride=2, padding=1)
leaky_relu(0.2)
flatten()
linear(1)
"""))
```

`hg.HNDL(file="disc.hndl")` loads a file; its text is copied into the recipe,
as TOML `file =` is today.

Any callable importable as `module:name` that returns an `nn.Module` is
accepted, including a plain factory function. A class defined in the script
being run (`__main__`) works for a foreground single-GPU run, because
`importlib.import_module("__main__")` resolves inside that process. A
background run, a multi-GPU run and a later resume from the CLI need an
importable module; the API refuses those cases up front and names the fix.

### 4. Watch metrics, samples and evaluations

```python
run = hg.load("runs/faces")          # finished, running, or on another machine

run.status, run.step, run.total_steps   # "running", 41200, 200000
run.metrics.names                       # ["loss/d_total", "diversity/ratio", ...]
d = run.metrics["loss/d_total"]         # Series: d.steps, d.values, d.latest
run.metrics.latest()                    # {"loss/d_total": 0.71, ...}
run.metrics.describe("loss/d_total")    # formula, units, owner, from the catalog
run.metrics.to_pandas()                 # optional; pandas is not a dependency

fid = run.evaluations["fid50k_train"]   # Series at source steps; status per point
run.previews[-1].image                  # PIL image of the latest EMA grid
run.previews.at(10_000).path            # nearest retained preview

for event in run.follow():              # blocks; yields new events as they commit
    if event.kind == "evaluation_complete":
        print(event.step, event.value)
```

All of this reads what the run already writes: `manifest.json`,
`events.jsonl`, `metrics/catalog-*.json`, `metrics/evaluations/*/` and
`previews/*/preview.json`. It reads without torch and without locks, as the
viewer does. `follow()` uses the reconnectable cursor that
`hypergan events --cursor` pages through. Long runs hold hundreds of
thousands of `train` events (the 200k CIFAR run has 204,237). `Run.metrics`
therefore reads from the existing `event_views` projection instead of
re-parsing JSONL each time.

Evaluations are declared on the recipe and scheduled as today:

```python
recipe = recipe.with_evaluation("fid", every=10_000, samples=50_000, device="cuda:1")
run.evaluate("fid", samples=50_000)     # explicit, on a stopped run
```

### 5. Run in the background (notebooks, scripts that do other work)

```python
run = hg.start("~/data/faces", recipe="image-64", run_dir="runs/faces")
run.wait(until_step=5_000)
run.previews[-1].image
run.checkpoint()        # save at the next complete update boundary
run.stop()              # cooperative stop; resumable
run = hg.resume("runs/faces", background=True)
```

`hg.start` launches `python -m hypergan train` in a subprocess with the
recipe written into the run directory. It then returns the same `Run` reader.
Stop, checkpoint and progress use the existing run signals and
`run_requests`, so a crash in a notebook kernel never takes the run with it.

### 6. More GPUs

```python
if __name__ == "__main__":
    run = hg.train("~/data/faces", recipe="image-64", run_dir="runs/faces", gpus=2)
```

`gpus=N` (or `gpus=[0, 1]`) selects the `cuda-replicated-nccl` profile over
those devices, with `CUDA_VISIBLE_DEVICES` set for the workers; `accumulate=K`
selects accumulation. `gpus=1` or omitted is native execution. Resume infers
the saved topology, as `hypergan resume` does now.

**Current capability, stated plainly:** the replicated trainer runs the toy
recipe today. The image recipes need independent phase draws, reused
components and prior bindings in the replicated loop, which the audit places
in 2.1. Until then `hg.train(..., gpus=2)` with an image recipe fails at
`recipe.check(gpus=2)` with that reason, rather than running something
different. The spawn workers are why the `__main__` guard is required. Multi-host
is a separate decision (below).

### 7. Use the trained generator

```python
g = hg.load("runs/faces").generator(device="cuda")   # EMA weights, from model.pt
images = g.sample(64, seed=3)          # tensor [64, 3, 64, 64] in [-1, 1]
z = g.latent(8, seed=3)                # draws from the learned particle prior
g.decode(z)                            # same images as g.sample(8, seed=3)
g.interpolate(z[0], z[1], steps=9)     # latent walk
g.particles                            # the learned prior table, [N, z_dim]
g.module                               # plain nn.Module for your own code
hg.save_grid(images, "grid.png")
g.export("faces.onnx")                 # later: ONNX with the prior as data
```

This wraps the existing verified inference bundle (`artifacts.sample`
already checks its hash and loads with `weights_only=True`). Conditional
generators take their inputs by keyword: `g.sample(8, condition=gray)`, as
`artifacts.sample(..., inputs=...)` does.

## Reference surface

Top level, all torch-free until they build or train:

| Name | Purpose |
| --- | --- |
| `hg.recipes()` | Catalog: name, task, resolution, qualification (measured metric, or "unmeasured") |
| `hg.recipe(name_or_path, **common)` | Load a packaged recipe or a TOML file as a `Recipe` |
| `hg.Recipe` | Immutable resolved configuration. `replace`, `with_evaluation`, `check(gpus=)`, `save`, `to_dict`, `describe` |
| `hg.Network(factory, **args)` / `hg.HNDL(source \| file=)` | Component specs; the recipe supplies I/O contracts |
| `hg.ImageFolder(path, size=, resize=)` | Explicit data spec; a plain path string is shorthand |
| `hg.train(data, recipe=, run_dir=, *, steps, gpus, accumulate, viewer, preview_every, checkpoint_every, max_seconds, on_event)` | Foreground run; returns `Run` |
| `hg.start(...)` | Same arguments; background subprocess; returns `Run` |
| `hg.resume(run_dir, *, checkpoint, recipe, background)` | Continue a run |
| `hg.load(run_dir)` | Read a run; no training |
| `Run` | `status`, `step`, `recipe`, `url`, `metrics`, `evaluations`, `previews`, `checkpoints`, `follow()`, `wait()`, `stop()`, `checkpoint()`, `evaluate()`, `generator()`, `sample()` |
| `Generator` | `sample`, `latent`, `decode`, `interpolate`, `particles`, `module`, `export` |

`run_dir` is optional. When it is omitted, runs go to `runs/<recipe>-<date>-<n>`
and the path is printed. `data` may be a path, an `hg.ImageFolder` or an
`hg.Network`-style importable data factory. The factory keeps today's contract,
`__call__(batch_size, *, generator) -> {"real": ...}`, and must be resumable.
`on_event` receives every committed event. It is an observer: it cannot change
training, because that would break the recipe-is-data rule.

The existing module-level functions (`hypergan.execution.train`,
`hypergan.artifacts.sample`, `hypergan.metric_evaluation.evaluate`) stay as
the implementation layer. They are not documented as public.

## Recipes to ship

The API is only as good as the recipes behind `hg.recipe(...)`. Every example
TOML on `develop` hard-codes a local data path and hash-pinned weights, and
`hypergan recipes` lists only the 2-D toy. The proposal is a small packaged
catalog with the data left as a slot:

| Name | Basis | Qualification |
| --- | --- | --- |
| `toy-2d` | `reference/100gaussians` | numerical reference (today) |
| `image-32` | CIFAR tiny-transformer + ResNet-feature discriminator | FID50k measured on CIFAR-10 (the 200k run) |
| `image-64`, `image-128` | the same generator family scaled, pixel discriminator | unmeasured until a run is published |

Pretrained discriminator weights stay explicit. A recipe that needs ResNet18
states the file and hash. `hg.fetch("resnet18")` or `hypergan fetch resnet18`
downloads and verifies it on request only, preserving the no-implicit-downloads
rule. Whether to ship scratch-discriminator variants that need no weights is
a decision below.

## Other things worth having, and where they belong

| Idea | Recommendation |
| --- | --- |
| Latent tools on the particle prior: nearest particle, image → latent projection, particle browsing | **2.0.** This is unique to ParticleGAN and costs little over `Generator`. |
| Run comparison: `hg.runs("runs/")` table of recipes, steps and final metrics | **2.0**, read-only, built on `hg.load` |
| Live notebook widgets (metric plot, latest grid) | 2.0 if cheap: `run.show()` in Jupyter renders the grid and a sparkline |
| Conditional recipes (colorization, class-conditional) | 2.1. The component bindings already support them; each needs a measured recipe |
| ONNX / TorchScript export | 2.0 per the audit, generator only, prior bundled |
| Hyperparameter sweeps | Out of scope. Document a loop over `hg.start` instead |
| Using HyperGAN's losses in your own training loop | Out of scope. That is ParticleGAN's API; say so in the docs |
| Multi-host training | 2.1+, separate decision below |

## Mapping onto existing code

| API | Built on | Gap to close |
| --- | --- | --- |
| `Recipe` | `config.resolve_config`, `config_values`, `network_config` | `prepare_train`/`run_train` take a `config_path` and re-read the file. They need to accept a resolved config, keeping `config_path` for the CLI and for relative `.hndl` paths. A recipe records its sources already materialized, so this is a small refactor. A `to_toml` writer is new. |
| `recipe.check()` | `model_description.build_networks` (meta device), `execution_preflight` | Contract checks for G/D roles and the BatchNorm rule |
| `hg.train` / `resume` | `execution.prepare_train`, `prepare_resume`, `web_autostart.training_viewer` | Viewer startup currently lives in `cli.py`; move it under the API |
| `hg.start`, `Run.stop/checkpoint` | subprocess CLI, `run_signals`, `run_requests` | A subprocess launcher and PID/attempt tracking |
| `Run` readers | `run_events.read_event_page`, `metrics.read_catalog`, `event_views.Projector`, preview manifests, evaluation receipts | Typed wrappers; `Series` type |
| `Generator` | `artifacts._sample` bundle loading | Split bundle loading from sampling; add `latent`/`decode`/`interpolate` |
| `gpus=` | `execution_profiles`, `replicated_execution` | Map N to a profile; capability check for image recipes |
| Recipe catalog | `examples/*.toml`, `config.list_recipes` | Package templates with a data slot; `fetch` |

## Implementation order

Each step is its own PR into `develop`, with tests in the fast suite. Steps
touching training paths also get a `heavy` end-to-end test on the toy recipe.

1. `Recipe`: load, replace, save and round-trip, torch-free. `prepare_train`
   accepts a resolved config. `hg.train` and `hg.resume` run in the foreground.
2. `Run` reader: status, metrics, previews, evaluations and `follow()`,
   torch-free.
3. `Generator`: sample, latent, decode, interpolate and `module`.
4. `hg.Network` / `hg.HNDL` with role contracts and `recipe.check()`.
5. Packaged `image-32` recipe with a data slot, `fetch`, and the
   `hypergan train FOLDER --recipe` shortcut.
6. `hg.start` background runs.
7. `gpus=` and `accumulate=` mapping with capability checks.
8. The first user tutorial, written against this API.

## Decisions for the owner

1. **Blocking default.** Should `hg.train` block by default, with `hg.start`
   for background runs? Recommended: yes. Scripts read naturally, and the
   notebook case has a name of its own.
2. **Networks defined in `__main__`.** Allow them for foreground single-GPU
   runs and refuse the rest with a clear message (recommended)? Or snapshot
   the script's source file into the run so background, multi-GPU and resume
   also work? Snapshotting is more convenient, but it copies arbitrary user
   code into run provenance.
3. **Which recipes ship in 2.0,** and does any of them need no pretrained
   weights? A weight-free `image-64` would make the first run a single
   command, possibly at a worse FID.
4. **Weights download on request.** Is `hg.fetch` / `hypergan fetch` an
   acceptable exception to the no-downloads rule, given it only runs when
   explicitly called?
5. **torch `Dataset` support.** Many users will have a `Dataset` rather than a
   folder. An adapter needs a seeded sampler and position state to keep exact
   resume. Is that in 2.0, or do we document the data-factory contract only?
6. **Multi-host.** Keep HyperGAN's own supervisor (coordinator-death recovery,
   exact resume) and add hosts to it? Or accept `torchrun`-launched workers and
   give up some recovery guarantees? This decides what `gpus=` grows into.
7. **Name and shape.** Plain functions plus a `Run` object, with no `Trainer`
   or `GAN` class (1.x had `hg.GAN`). Recommended, because the run directory,
   not an in-memory object, is the unit users keep, resume and share.
