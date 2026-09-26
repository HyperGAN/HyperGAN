# API POC A: config first

Branch `poc/api-config-first` (off develop `bff6de3d`). Stance: the model file is
the source of truth. It is a TOML file (plus `.hndl` files) where networks
declare a **role** and every loss is **listed together**. The Python API is a
small set of plain functions over that file. User Python is named by import
path (`module:object`), never copied. Everything lowers, without Torch, to the
recipe `resolve_config` already takes. That keeps fingerprints, resume, the
viewer, the CLI and replicated workers unchanged.

## The API on one page

```python
import hypergan.api as hg

# model files: the thing people edit, share and review
model = hg.load("model.toml")                 # or hg.packaged("gaussian-grid"), hg.from_dict({...}, base=dir)
model = hg.override(model, {"train.steps": 200, "losses.reconstruction.weight": 0.5})   # returns a copy
hg.validate(model)      # torch-free; raises with the field named, returns warnings
hg.lower(model)         # the engine recipe (what the run records)
hg.fingerprint(model)   # numerical identity; equal => same recipe
hg.save(model, "copy/model.toml")             # TOML; .hndl files referenced, not copied

# building in Python: helpers return plain dicts shaped like the file
hg.hndl(source | file=, role=, inputs=, input_shape=, output_shape=)
hg.module("pkg.nets:Generator", role=, inputs=, **args)
hg.adversarial(critic, weight=, penalty=)  hg.reconstruction(fn=, input=, target=)
hg.objective(id, fn, inputs=)              hg.prior_loss(weight)

# training
run = hg.train(model, "runs/x", previews=50, profile=None)   # profile="cpu-replicated-gloo" etc.
run = hg.resume(run)                                          # or hg.train(model, same dir) to continue

# reading a run (needs only the run directory)
run = hg.open_run("runs/x")         # Run(path, manifest): .status .step .config .fingerprint
hg.metrics(run)                     # {"loss/d_total": [(step, value), ...], "d_over_g": [...]}
hg.evaluations(run)                 # {"modes": [{"step", "value", "status", ...}]}
hg.evaluate(run, "nearest")         # run a manual evaluation now
hg.samples(run)                     # periodic EMA previews as Sample(step, data, inputs, ...)
hg.sample(run, count=64, seed=1)    # fresh samples from the final EMA model
hg.view(sample, "scatter", run=run) # apply a sampler from the model file -> Image / Audio / anything
```

Data objects: `Model(spec, base, path)`, `Run(path, manifest)`, `Sample`,
`Image`, `Audio`. There is no trainer or GAN class. The CLI takes the same file:
`hypergan validate|train|resume|evaluate` all accept a model file, because
`load_config` detects `[networks]` and lowers it.

### The model file

```toml
format = "hypergan-model/1"
name = "demo/encoder-two-critics"

[data]                                   # builtin = "..." | factory = "m:f" (batch) | dataset = "m:C" (item)
dataset = "toy_project:PairedVectors"
args = { size = 512, dims = 2 }

[prior]
kind = "particles"                       # other keys are prior args
z_dim = 4
num_particles = 64

[networks.encoder]
role = "encoder"                         # generator | critic | encoder | auxiliary
source = "linear(8)\nleaky_relu(0.2)\nlinear()"
input_shape = ["B", 2]
output_shape = ["B", 2]
inputs = { input = "batch.condition" }

[networks.decoder]
role = "generator"
source = """
concat(x, condition)
linear(16)
leaky_relu(0.2)
linear(16)
leaky_relu(0.2)
linear()
"""
input_shape = { x = ["B", 4], condition = ["B", 2] }
output_shape = ["B", 2]
inputs = { x = "latent", condition = "encoder" }     # a network is bound by its name

[networks.pair_critic]
role = "critic"
source = "concat(x, condition)\nlinear(16)\nleaky_relu(0.2)\nlinear()"
input_shape = { x = ["B", 2], condition = ["B", 2] }
output_shape = ["B", 1]
inputs = { x = "candidate", condition = "batch.condition" }

[networks.marginal_critic]
role = "critic"
hndl = "critic.hndl"                     # file next to the model file (module = "m:Class" also works)
input_shape = ["B", 2]
output_shape = ["B", 1]
inputs = { x = "candidate" }

# Everything the model optimizes, in one place.
[[losses]]
type = "adversarial"                     # RpGAN logistic + K3P; penalty = coefficient
critic = "pair_critic"
penalty = 1.0

[[losses]]
type = "adversarial"
critic = "marginal_critic"
weight = 0.5
penalty = 0.5

[[losses]]
type = "reconstruction"                  # or "objective" with inputs = {...}
id = "reconstruction"                    # publishes loss/objectives/reconstruction
fn = "l1"                                # mse | l1 | module:object
input = "generated"
target = "batch.real"

[[losses]]
type = "prior"                           # particle-table spread regularizer

[penalty]                                # shared K3P settings (kappa, lazy_k, anchor_*)
kappa = 1.0

[train]                                  # optimizer + schedule fields in one table
steps = 12
batch_size = 16
device = "cpu"

[samplers.scatter]                       # read-side: sample -> viewable
fn = "toy_project:scatter"

[metrics.custom.d_over_g]                # cheap: function of update scalars
fn = "toy_project:d_over_g"
inputs = { d = "d_loss", g = "g_loss" }

[evaluations.modes]                      # expensive: EMA snapshot, own (holdout) data
fn = "toy_project:modes_covered"
every = 1000                             # omit for manual-only
device = "cpu"
samples = 256
batch_size = 64
seed = 7
data = { dataset = "toy_project:GridPoints", args = { split = "holdout" } }
```

Bindings are `latent`, `batch.<field>`, `generated`, `candidate`, `prior.means`,
or a network name (`encoder`, `encoder.<output>`). Unknown fields, missing roles,
unused critics, unknown bindings and engine limits fail with the field named
(tests cover ten such errors). A learnable prior with no `type = "prior"` loss
gets weight 0 plus a warning: a loss that is not listed is not trained.

User code is ordinary: an `nn.Module`, a dataset with `__len__`/`__getitem__`
(or `load(i)`), and functions (`toy_project.py` in the examples has no HyperGAN
imports except the `hg.image` helper).

## What ran (CPU, `particlegan08-env`, torch 2.14, ParticleGAN 0.8.0, HNDL 0.8.0)

All scripts are in `examples/api/config_first/`. Every run below finished `complete`. The output is copied from the runs.

**1. Simplest case** (`demo_1_simple.py`, `simple.toml`): generator is
`toy_project:MLPGenerator` (an `nn.Module`), critic is `critic.hndl`, built-in
2-D Gaussian grid, 20 steps, previews every 5.

```python
model = hg.load(HERE / 'simple.toml')
run = hg.train(model, run_dir, previews=5)
series = hg.metrics(run)
```
```
status=complete steps=20 fingerprint=6beaab2cfe66
loss/d_total                 step  20  0.6767
loss/g_total                 step  20  0.8122
loss/gradient_penalty        step  20  0.0076
loss/prior_regularizer       step  20  0.0866
diversity/ratio              step   5  0.1028
```

**2. Multi-network** (`demo_2_multi_network.py`, `two_critics.toml`, shown above):
encoder, generator (named `decoder`), two critics, L1 reconstruction and the prior
loss, on an item-level synthetic dataset. The script lowers and validates the
model, then trains 12 steps. Lowered recipe (excerpt):

```
components: {'encoder': {'input': 'batch.condition'}, 'generator': {'x': 'latent', 'condition': 'components.encoder'},
             'discriminator': {'x': 'candidate', 'condition': 'batch.condition'}, 'marginal_critic': {'x': 'candidate'}}
adversarial_terms: [{"id": "marginal_critic", "component": "marginal_critic", "weight": 0.5, "penalty": true,
                     "real": "batch.real", "fake": "generated", "penalty_coeff": 0.5}]
objectives: [{"id": "reconstruction", "factory": "l1", "inputs": {"input": "generated", "target": "batch.real"}, ...}]
status=complete steps=12 fingerprint=c6ba729ecc82
loss/d_total 1.0000  loss/g_total 2.7833  loss/gradient_penalty 0.0109
loss/objectives/reconstruction 1.2963  loss/prior_regularizer 0.3519  (step 12)
```

The same model built in Python (from `demo_4_round_trip.py`; it has the same fingerprint as the file):

```python
hg.from_dict({
    'name': 'demo/encoder-two-critics',
    'data': {'dataset': 'toy_project:PairedVectors', 'args': {'size': 512, 'dims': 2}},
    'prior': {'kind': 'particles', 'z_dim': 4, 'num_particles': 64},
    'networks': {
        'encoder': hg.hndl('linear(8)\nleaky_relu(0.2)\nlinear()', role='encoder',
                           input_shape=['B', 2], output_shape=['B', 2], inputs={'input': 'batch.condition'}),
        'decoder': hg.hndl(DECODER, role='generator', input_shape={'x': ['B', 4], 'condition': ['B', 2]},
                           output_shape=['B', 2], inputs={'x': 'latent', 'condition': 'encoder'}),
        'pair_critic': hg.hndl(PAIR, role='critic', input_shape={'x': ['B', 2], 'condition': ['B', 2]},
                               output_shape=['B', 1], inputs={'x': 'candidate', 'condition': 'batch.condition'}),
        'marginal_critic': hg.hndl(file='critic.hndl', role='critic', input_shape=['B', 2],
                                   output_shape=['B', 1], inputs={'x': 'candidate'}),
    },
    'losses': [hg.adversarial('pair_critic', penalty=1.0),
               hg.adversarial('marginal_critic', weight=0.5, penalty=0.5),
               hg.reconstruction(fn='l1'), hg.prior_loss(1.0)],
    'penalty': {'kappa': 1.0, 'lazy_k': 1},
    'train': {'steps': 12, 'batch_size': 16, 'seed': 42, 'device': 'cpu'},
    'sampling': {'count': 8, 'seed': 123},
}, base=HERE)
```

**3. Extension points** (`demo_3_extensions.py`, `extensions.toml`), 20 steps:

| Extension | User writes | How it is wired | Ran |
|---|---|---|---|
| Data loader | `GridPoints`: `__len__`, `__getitem__` → `{"real": tensor}` | `hypergan.item_data:ItemData` (new) is the engine's batch factory. HyperGAN owns order, shuffle, epochs, collation (nested dicts/lists), resume state, rollback on a failed read | yes, training and evaluation data; a test checks resume matches a straight run exactly |
| Metric | `d_over_g(d, g)` | `plugin_functions:ScalarFunction` adapts it to the existing bounded scalar-metric workers; the function's source hash is in its definition | yes: `d_over_g` step 10 = 0.7189 (step 20 was dropped: earlier call still outstanding) |
| Evaluation | `modes_covered(generated, reference)` on holdout items | `plugin_functions:SnapshotFunction` on the existing snapshot evaluator (interval and manual) | yes: `modes` step 10 = 0.12 (interval); `nearest` step 20 = 0.1447 (manual `hg.evaluate`) |
| Sampler | `scatter(sample) -> hg.image(...)` | declared in `[samplers]`, recorded with the run (not fingerprinted), applied by `hg.view` at read time | yes: 64×64 PNG from a preview and from `hg.sample` |

**4. Config round trip** (`demo_4_round_trip.py`): four routes, one fingerprint.
Changing one loss weight changes it.

```
file                             c6ba729ecc82bb4b10fda68b3355c4e0db67352eaac3e9cd8fbc2ab9c0b5a4f5
file saved + reloaded            c6ba729ecc82bb4b10fda68b3355c4e0db67352eaac3e9cd8fbc2ab9c0b5a4f5
python-built saved + reloaded    c6ba729ecc82bb4b10fda68b3355c4e0db67352eaac3e9cd8fbc2ab9c0b5a4f5
CLI loader (load_config)         c6ba729ecc82bb4b10fda68b3355c4e0db67352eaac3e9cd8fbc2ab9c0b5a4f5
override losses.reconstruction.weight=0.25 -> 99f6c15ab059c862 (differs)
```

The packaged `hg.packaged("gaussian-grid")` model file lowers to exactly
`hypergan.config.DEFAULT`, with the same fingerprint and `recipe_match = True`. TOML is kept
because the owner's tooling already reads it. `hypergan.toml_writer` is a
small writer (no `tomli_w` needed), and a test runs every `examples/*.toml` through it without changes.

**5. Reading a run** (`demo_5_read_run.py`, on demo 3's run directory):

```
run run: status=complete steps=20
fingerprint matches model file: True
18 metric series; last values:
  loss/d_total                   step  20  0.6626
  loss/g_total                   step  20  0.8266
  d_over_g                       step  10  0.7189
  throughput/steps_per_second    step  20  104.3187
  evaluation modes              step 10: 0.1200 (complete)
  evaluation nearest            step 20: 0.1447 (complete)
  preview step 5 name=g data=(16, 2) unique particles=16
  scatter of last preview -> preview-scatter.png
  fresh EMA sample step 20: (32, 2) mean=[-0.0886, -0.1955]
  scatter of fresh sample -> fresh-scatter.png
```

**Multi-process** (`demo_replicated.py`): the one-critic model with the item
dataset, `profile="cpu-replicated-gloo"`, 2 ranks, 6 steps. The run completed
with `global_batch_size 64, local_batch_size 32`, and
`loss/d_total [0.6897, 0.674, 0.6682, 0.6512, 0.6502, 0.6545]`. Each rank rebuilt
the `nn.Module` generator and `ItemData` from the recorded recipe.

## Tests

- `tests/foundation/test_api_poc_config_first.py`: 51 fast tests pass in 2.5 s. They cover: writer round trips over every example, lowering, the packaged reference fingerprint, save/load/Python-built fingerprints, override by loss id, 10 named errors, samplers not in the fingerprint, metric/evaluation lowering, the item-data resume/rollback/nested collation, the adapters, training plus metric reading plus sampling, stop-and-continue with item data matching an uninterrupted run, and training the multi-network model. One `heavy` test (metric and evaluation workers, previews, samplers) passes in 10.8 s with `-m heavy`.
- Fast foundation suite: 783 passed, 16 failed. The same 16 fail on the base commit exported from `bff6de3d` (plus one provenance test there, because an export has no git metadata). All of them are `python -I` subprocesses that cannot import a PYTHONPATH checkout, plus the wheel-only distribution test. This is the known environment issue in the CPU-suite memory note. None is caused by this branch.

## Changes to existing code (small, separate)

- `config.load_config` also accepts an in-memory mapping (a raw recipe, a resolved config, or a model file), and lowers model files it reads from disk. Existing recipes resolve exactly as before; a test checks `paired-linear.toml`.
- `config.resolve_config` accepts an optional top-level `samplers` table, handled like `adversarial_terms`: when absent it is not stored, so legacy fingerprints do not change. `config_values` records it with the run. `numerical_values` and `fingerprint` exclude it, because it is an observation setting.
- `execution.prepare_train` names `resume --config` as the remedy when only `metrics`/`samplers` changed.
- `pyproject.toml` adds `models/*.toml` to package data.
- New modules: `api.py`, `model_file.py`, `toml_writer.py`, `item_data.py`, `plugin_functions.py`, `models/gaussian-grid.toml`.

## Gaps (honest)

1. **Engine limits the file cannot hide.** These are rejected with messages rather than silently changed:
   - exactly one `role = "generator"`;
   - the first adversarial loss must compare `batch.real` with `generated` through its critic's own inputs; later adversarial losses are free;
   - one critic optimizer and one generator-side optimizer, so an `encoder` trains with the generator;
   - schedule `d-then-g-v1` only;
   - replicated execution rejects a second adversarial loss.
   There is no per-network optimizer or schedule section yet. `[train]` holds the one set.
2. **No per-term series for extra critics.** Their losses are summed into `loss/d_adversarial` and the D/G totals. The loss list already gives each term an `id`, so `loss/adversarial/<id>` is a natural engine addition.
3. **Custom metrics bind only `update.*` scalars.** Objectives are already published as `loss/objectives/<id>`, but a user function cannot combine them yet. Each call starts a fresh worker with one outstanding call per metric, so fast loops drop calls (seen at step 20).
4. **Samplers are read-side only.** `hg.view` applies them. The preview worker and the viewer do not run them yet, so for non-image output the viewer still shows numeric JSON. The natural next step is for the preview worker (already a separate CPU process with an EMA copy) to run the declared samplers and publish PNG/WAV.
5. **Evaluation data must be flat batched tensors with `real`** (engine worker check). Nested items work for training data but not for evaluation data. Previews and evaluations need one tensor, chosen with `sampling.output` / `evaluations.<id>.output`. Nested generator outputs are reachable by dotted bindings.
6. **Names are lowered.** The generator network becomes `generator` and the first critic becomes `discriminator` in the recorded recipe, so the viewer and model description show those names. A recorded name map would fix this.
7. **Provenance of the file.** The run records the lowered recipe, which contains the HNDL text, as today. It does not record the model-file path or hash. Recording `{path, sha256}` (no copy) would link a run to the file that made it.
8. **Import paths must resolve in every process.** User code must be an installed package or on the path. Spawned workers inherit `sys.path`, which is how the example directory works. `hypergan train model.toml` from the CLI needs `PYTHONPATH` for a loose `toy_project.py`. The loader does not change `sys.path`.
9. **`hg.save` drops comments.** It writes the model as data. That is right for models built in Python and for copies, but a load → override → save cycle on a hand-edited file loses its comments. `override` plus `train` never rewrites the file. A `--set key=value` CLI flag would be the file-preserving equivalent (not built).
10. **Not demonstrated:** LoRA-weight or latent-code outputs, audio or video models (an `hg.audio` helper and a `tone` sampler exist but are untested on a real model), and CUDA (the owner is training).

## How multi-host fits

- **Same file, same meaning.** `model_file.lower` is a pure function of the file bytes and the referenced `.hndl` bytes. It imports no Torch and no user code. The run records the lowered recipe, and replicated workers already receive that recipe (`config_values`), not the file. A host never needs the model file, only the same importable code. Preflight already compares construction identity across ranks.
- **HyperGAN shards data.** `ItemData` draws one epoch seed per epoch from the engine's data stream, so the global index order is identical on every host. Its checkpoint state is three integers (epoch, cursor, seed), not a permutation. Today the engine draws the global batch on every rank and slices it (`replicated-global-draw-rank-slice`), so each rank loads every item of the global batch. `ItemData.indices()` exposes the global order, so a later engine can have each rank or host load only its slice without changing model files, order or resume state.
- **Nothing is copied.** Workers import user code by path, which is the rebuild-from-config-plus-code rule; a test asserts that the run directory contains no `.py` or `.hndl` files.

## Assessment of this stance

Strengths: one reviewable file per model, which the CLI, the Python API and the
viewer all read identically. Roles and a single loss list make a model's
objective readable at a glance: the two-critic model is 91 lines of TOML (comments included) with
every trained term visible. The Python layer is 27 small functions (including builders and viewable helpers) with no
object model, and Python-only users get the same file through `from_dict` and `save`.

Costs: the file schema is another surface to design and document. Its
vocabulary (`role`, `type`, bindings) must track the engine, and each engine
limit shows up as a lowering error rather than a missing method. A model that
needs control flow or loops (for example, generated network lists) needs
Python that builds the dict. That works, but the file then becomes an output
rather than something written by hand. Custom user logic only enters through
named extension points (network, data, loss `fn`, metric, evaluation,
sampler). This is deliberate, and it is why every process can rebuild the model.
