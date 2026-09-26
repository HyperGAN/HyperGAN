# API POC B: functional graph builder (`hypergan.graph`)

Branch `poc/api-graph`, 2026-09-26. Stance: you build the model in Python from
references and pass **one list of losses**. Each network's role comes from those
losses. The graph becomes the existing config tables and is saved as a TOML file.
That file is the durable artifact: `hypergan train model.toml` runs it unchanged.
No user code is copied. Everything is referenced by `module:object`.

Code: `src/hypergan/graph/` (about 900 lines). Demos: `examples/api/graph/`.
Tests: `tests/foundation/test_api_poc_graph.py`.

## The API on one page

```python
import hypergan.graph as hg

# References: values in the graph (nothing runs yet)
x    = hg.data("real")                  # a batch field; hg.data("condition"), hg.data("x", shape=["B", 2])
z    = hg.latent(64)                    # the one prior (particles; kind="mog"/"gaussian")
hg.candidate                            # what a critic scores
net  = hg.net(MyModule, x=x, hidden=64) # reference kwargs are forward inputs, the rest are constructor args
net  = hg.hndl(source_or_None, file="g.hndl", shape=["B", 2], x=z, name="g2")   # input shapes are inferred
out  = net["mu"]                        # a nested output (dict key or list index)
twin = net.reuse(name="twin", x=other)  # the same weights on other inputs

# Losses, listed together. Roles come from these.
hg.adversarial(critic, real=x, fake=g, weight=1.0, penalty=None, name=None)
hg.l1(a, b, weight=1.0, name=None)      # also hg.mse; the target is detached by default
hg.loss(my_fn_or_class, weight=0.1, input=g, radius=1.0)   # a custom generator-side loss

# Observation: cheap metrics, expensive evaluations, what people look at
hg.metric(fn, every=10)                 # fn(g_loss, d_loss): the parameter names pick update scalars
hg.evaluation(fn, holdout_data, every=5000, samples=256)   # fn(generated, reference) -> float
hg.sampler(fn, points=g, count=64)      # fn(**outputs) -> hg.view.image/points/audio/text/tensor

# Data: HyperGAN owns batching, order, seed, resume position and (later) sharding
hg.dataset(MyItems, count=512)          # MyItems has __len__/__getitem__ only
hg.batches("pkg:Factory", **args)       # the existing batch contract; hg.gaussian_grid(), hg.paired_linear()

# Model, file, training, reading
m   = hg.model(data, losses, metrics=[...], evaluations=[...], samplers=[...],
               steps=..., batch_size=..., device=..., settings={"optimizer": {"lr": 1e-4}})
m   = hg.model(data, generator=G, discriminator=D, latent=4)          # shorthand for one pair
hg.save(m, "model.toml"); m = hg.load("model.toml"); hg.fingerprint(m); print(hg.describe(m))
run = hg.train(data, losses, run="runs/x", steps=...)   # or hg.train(m, run=...) / hg.train("model.toml", run=...)
hg.metrics(run); hg.last(run); hg.evaluations(run); hg.evaluate(run, "fid"); hg.samples(run); hg.previews(run)
```

How the graph becomes config:

- **Roles.** The first `hg.adversarial` is the main pair. Its `fake` network becomes
  the engine's `generator`. Its critic becomes `discriminator`. Every other network
  scored by an adversarial loss is a critic, and its term goes into `[[adversarial_terms]]`.
  A network upstream of a fake sample or a generator-side loss trains with the generator.
  A network that only feeds critic conditioning, samplers or evaluations is set
  `trainable = false`, and `describe()` reports it. `hg.roles(config)` gives the same
  answer from any config file.
- **Stable names.** Names are checkpoint keys and metric ids, so call order never
  decides them. The main pair is always `generator`/`discriminator`. Any other
  PyTorch network is named after its class in snake_case (`Encoder` becomes `encoder`).
  An HNDL network that is not in the main pair needs `name=`. If two networks would
  get the same name, the build fails and asks for `name=`. Loss ids work the same
  way: `name=`, otherwise `l1`, `mse` or the function name, and a clash fails.
  Metric ids come from these, e.g. `loss/objectives/reconstruction`.
- **Import paths only.** A class defined in `__main__` or inside a function is
  refused, with a message saying to move it into a module. Every process rebuilds
  the model from the file plus importable code. Constructor arguments must be plain
  data: no None, NaN or objects.
- **The file.** `hg.save` writes TOML with a small writer, `graph/toml_writer.py`
  (tomllib reads but cannot write, and tomli_w is not installed). `.hndl` files are
  referenced by a relative path, not inlined. The header records a hash of the body.
  Saving again replaces the file only if nobody has edited it since. After a hand
  edit, `hg.save` refuses and says to use `hg.load`. This guards the "user may be
  editing the wrong one" concern: the Python and the file cannot silently diverge.

## The multi-network model in this API (demo 2, `multi_network.py`)

```python
x, c, z = hg.data(), hg.data("condition"), hg.latent(4)
code   = hg.net(Encoder, x=c, code=4)                        # returns {"mu", "scale"}
fake   = hg.net(ConditionalGenerator, z=z, code=code["mu"])
recon  = hg.net(Decoder, code=code["mu"])
pair   = hg.net(PairCritic, x=hg.candidate, condition=c)
marginal = hg.hndl(file="critic.hndl", shape=["B", 1], x=hg.candidate, name="marginal")
model = hg.model(hg.paired_linear(), [
    hg.adversarial(pair, real=x, fake=fake),
    hg.adversarial(marginal, real=x, fake=fake, weight=0.5, penalty=1.0),
    hg.l1(recon, c, name="reconstruction"),
    hg.mse(fake, x, weight=0.1, name="paired"),
], steps=10, batch_size=16, device="cpu")
```

The saved file (`examples/api/graph/multi_network.toml`, abridged):

```toml
[components.generator]
factory = "my_project:ConditionalGenerator"
inputs = { z = "latent", code = "components.encoder.mu" }
[components.discriminator]
factory = "my_project:PairCritic"
inputs = { x = "candidate", condition = "batch.condition" }
[components.encoder]
factory = "my_project:Encoder"
args = { code = 4 }
inputs = { x = "batch.condition" }
[components.marginal]
factory = "hndl"
args = { file = "critic.hndl", output_shape = ["B", 1], input_shape = ["B", 2] }
inputs = { x = "candidate" }
[components.decoder]
factory = "my_project:Decoder"
inputs = { code = "components.encoder.mu" }
[[objectives]]
id = "reconstruction"
factory = "l1"
inputs = { input = "components.decoder", target = "batch.condition" }
[[adversarial_terms]]
id = "marginal"
component = "marginal"
weight = 0.5
real = "batch.real"
fake = "generated"
penalty = true
penalty_coeff = 1.0
```

## The demos: code and what ran

Every demo ran on CPU (`CUDA_VISIBLE_DEVICES=""`), with the owner's training
environment (torch 2.14, particlegan 0.8.0, hndl 0.8.0) and `PYTHONPATH` set to
this worktree. Command: `cd examples/api/graph && python <demo>.py <runs-dir>`.

**1. The simplest case (`simple.py`).** A PyTorch generator class and an HNDL-file
critic on the 2-D Gaussian grid, using the shorthand:

```python
run = hg.train(hg.gaussian_grid(), generator=Generator,
               discriminator=hg.hndl(file="critic.hndl", shape=["B", 1], x=hg.candidate),
               latent=4, run=runs / "simple", steps=40, batch_size=64, device="cpu")
print(hg.last(run, ["loss/d_total", "loss/g_total"]))
```

- Ran 40 steps in about 2 s.
- Final values: `loss/d_total 0.6845`, `loss/g_total 0.7760`, `loss/gradient_penalty 0.0014`,
  `loss/prior_regularizer 0.0685`.
- The demo asserts that the explicit form (`hg.model(..., [hg.adversarial(d, real=x, fake=fake)])`)
  has the same fingerprint as the run.
- A test also shows that the shorthand with the packaged HNDL source reproduces the
  built-in reference recipe's fingerprint exactly.

**2. Multi-network (`multi_network.py`).** The code is the model shown above.

- Validated and trained 10 steps in about 3 s. Final values: `loss/d_total 0.9236`,
  `loss/g_total 2.8638`, `loss/objectives/reconstruction 0.8845`, `loss/objectives/paired 0.4717`,
  `loss/gradient_penalty 0.0080`.
- `hg.describe` printed these roles: generator, discriminator (critic), encoder
  (generator-side), marginal (critic), decoder (generator-side).
- `python -m hypergan validate multi_network.toml` and
  `python -m hypergan train multi_network.toml --run-dir ...` also ran (10 steps).
  The CLI run's `config_sha256` was `992b9818…`, the same as the Python-built model.

**3. Extension points (`extensions.py`).**

```python
x = hg.data("x", shape=["B", 2]); fake = hg.net(Generator, z=hg.latent(4), hidden=32)
d = hg.hndl(file="critic.hndl", shape=["B", 1], x=hg.candidate)
model = hg.model(hg.dataset(RingItems, count=512),                     # item-level loader
    [hg.adversarial(d, real=x, fake=fake), hg.loss(radius_loss, input=fake, weight=0.1, radius=1.0)],
    metrics=[hg.metric(g_over_d, every=10, label="G/D loss ratio")],    # def g_over_d(g_loss, d_loss)
    evaluations=[hg.evaluation(ring_error, hg.dataset(RingItems, count=256, seed=1), every=50,
                               samples=128, batch_size=32, device="cpu")],
    samplers=[hg.sampler(scatter, points=fake, count=128)],             # -> hg.view.points
    steps=200, batch_size=32, device="cpu")
run = hg.train(model, run=runs / "extensions", preview_every=50)
```

- Ran 200 steps in about 12 s. Everything was wired to the existing engine.
- **Data:** `RingItems` implements only `__len__`/`__getitem__`. The `ItemDataset`
  adapter draws one permutation per epoch from the run's data stream and batches it.
  It saves epoch and position in checkpoints (`resume_supported: true`), renames the
  item field `x` to the engine's `real`, and records its identity plus the
  dataset source SHA-256.
- **Loss:** `loss/objectives/radius_loss` was published every step (last value 0.0236).
- **Metric:** `g_over_d` ran in the engine's plugin worker. It published one value
  (step 10: 1.468) in 200 steps. See the gaps for why.
- **Evaluation:** `ring_error` ran in the engine's snapshot worker on holdout items:
  0.827 at step 50.
- **Sampler:** `hg.samples` wrote `samplers/scatter/step-00000200/scatter.svg` and `.json`.

**4. Config round trip (`round_trip.py`).**

```python
path = hg.save(build(steps=3), runs / "round_trip.toml", overwrite=True)
assert hg.fingerprint(model) == hg.fingerprint(hg.load(path)) == fingerprint(load_config(path))
```

- The printed fingerprints were identical for the built model, `hg.load` and the CLI
  loader (`0d2ea06e…`). Roles were identical too.
- After a hand edit (`weight = 0.1` changed to `0.2`), `hg.save` refused: "was edited after
  hg.save wrote it; load it with hg.load(path)...".
- `hg.train(path, run=...)` then trained the edited file for 3 steps. Its manifest
  fingerprint matched the edited file (`65a8ef4d…`).

**5. Reading a run (`read_run.py`).**

```python
run = hg.Run(runs / "extensions")
hg.metrics(run)                          # {id: [(step, value)]}, incl. custom scalars
hg.evaluate(run, "ring_error")           # explicit snapshot evaluation now
hg.evaluations(run)                      # {id: [{step, value, status, evaluation_id}]}
hg.samples(run, count=32, seed=7)        # re-render samplers from the EMA bundle
hg.previews(run)                         # preview.json paths published during training
```

- Output: `loss/d_total` had 200 points (last 0.6406). The explicit evaluation completed
  with value 0.6099. The evaluations were step 50: 0.827 and step 200: 0.610.
  The sampler wrote `scatter.svg`/`.json`, and there was 1 preview.

**6. Multi-process (`replicated.py`, extra).** The item-level model trained with
`profile="cpu-replicated-gloo"`: 2 ranks, global batch 32, local batch 16, 20 steps,
about 7 s. Final values: `loss/d_total 0.6102`, `loss/g_total 0.9444`. Each rank rebuilt the
model from the config tables and `my_project`.

## Tests

- `test_api_poc_graph.py`: 13 fast tests plus 1 `heavy` test (the custom metric and
  evaluation start worker processes). All 14 pass (`-m ""`, 6.3 s). They cover:
  - the TOML writer round trip, including the full default config and awkward strings
  - lowering, and shorthand-explicit equality
  - reproducing the reference recipe
  - roles, name clashes, and freezing of conditioning-only networks
  - rejecting `__main__`/local classes and None arguments
  - the one-prior limit, and the real-field contract
  - save/load/CLI fingerprint equality, and refusing edited files
  - samplers staying out of the fingerprint
  - `ItemDataset` epochs and resume position
  - two end-to-end trainings, one of them sampling nested encoder outputs
- Fast foundation suite: 745 passed, 16 failed. A pristine export of `origin/develop`
  gives 731 passed, 17 failed. The same 16 fail in both; they are subprocess tests
  using `python -I`, which cannot see `hypergan` in this environment, plus the
  wheel-only distribution test. The develop export fails one more,
  `test_provenance`, only because the export is not a git checkout.
- `tests/reference`: 583 passed, 1 failed. The failure is the same `-I` subprocess
  issue (`test_data_cli`).

## Engine changes (small and separate: commit c53cb0c6)

- `config.py`:
  - An optional top-level `[samplers.<name>]` table (`factory`, `args`, `inputs`, `count`,
    `seed`), validated without importing torch. Candidate bindings are refused.
  - Samplers are observation settings, like `metrics`. `config_values` records them, so
    the manifest and the inference bundle carry them. `numerical_values` does not, so
    fingerprints are unchanged.
  - `load_config` now delegates its `file`/`network_files` inlining to a new
    `resolve_config_at(raw, base)`, so a Python-held raw dict resolves exactly as the
    file does.
- `artifacts.py`:
  - The inference-bundle loader is split out as `load_inference(run_dir)`, which
    `sample()` still uses.
  - `bundle_state` keeps the components and batch inputs that samplers bind.

Existing behavior and fingerprints are unchanged; the suites above show this.

## Gaps (honest)

1. **The main pair's names are fixed.** The engine requires components called `generator`
   and `discriminator`. The first adversarial loss's fake must be a whole network output,
   and its real must be a data field, which becomes `batch.real`.
2. **Engine limits, exposed as-is.** There is one prior (a second `hg.latent` fails the
   build). There is one critic optimizer and one generator-side optimizer, and only the
   `d-then-g-v1` schedule. You cannot give the encoder its own optimizer.
3. **Replicated execution rejects extra adversarial terms.** Demo 2 runs single-process
   only; demo 6 uses one critic.
4. **Extra critics have no per-term metrics.** Only aggregated `loss/*_adversarial` and
   `loss/gradient_penalty` are published. A `loss/adversarial/<id>` series needs an engine change.
5. **"Cheap" custom metrics are not cheap yet.** A plain-function metric runs in a fresh
   plugin worker per call, and a busy metric drops observations: `every=10` over 200 steps
   published once. Such metrics can bind only `update.*` scalars, not objective
   contributions. An in-process path for plain functions needs an engine change.
6. **Samplers run after training.** `hg.samples(run)` runs on the final EMA bundle, not
   during training, and the viewer does not show sampler output yet. Live previews stay
   image- and tensor-oriented. Wiring samplers into `preview_worker` is the next step.
7. **The item loader is basic.** It is synchronous (optional thread `workers`, no
   prefetch), and it converts float64 to float32. In replicated execution, every rank
   still loads the whole global batch and then slices it. `ItemDataset.plan`/`load` are
   split so a rank can load only its slice, but the replicated loop does not call them yet.
8. **Provenance of user functions is partial.** Function losses are hashed through the
   adapter module; the user module is not in the implementation hash. Metric and
   evaluation descriptions, and dataset identity, include the user source SHA-256.
9. **Scripts need a main guard.** Plugin, evaluation and replicated workers are spawned
   processes that import the script (the demos use `if __name__ == "__main__":`).
10. **HNDL shape inference is limited.** It needs known input shapes: latent, builtin
    data, `hg.data(..., shape=)` or other HNDL outputs. A PyTorch network feeding an HNDL
    network needs `input_shape=`.
11. **The file loads as tables, not a graph.** `hg.load` returns the config tables and
    their roles, not Python references. Iteration happens in the Python or in the file,
    and the edit guard stops the two from silently diverging.
12. **Some file paths are absolute.** An `.hndl` file far from the config is written with
    an absolute path, which is not portable across hosts.
13. **Longer training needs a constant learning rate.** Extending `steps` on an existing
    run works only when the learning rate is constant (`lr_floor = 1`); this is engine
    resume policy.

## How multi-host fits

The contract is the file plus import paths. No closures or pickled objects are
involved, and the replicated demo already runs this way. Several properties carry
over to many hosts:

- **Same model everywhere.** Lowering is deterministic, and names never depend on call
  order. A script run on every host therefore writes byte-identical TOML with the same
  fingerprint. A launcher can equally ship only `model.toml` and run
  `hypergan train model.toml` on each host.
- **HyperGAN owns data order.** The item plan is a pure function of the data seed and the
  saved position, so every host can compute the global plan. Each host then loads only
  its slice through `ItemDataset.load(plan[rank])`. That is one change in the replicated
  loop, and user dataset code does not change.
- **Snapshot-based observation.** Evaluations and samplers already run from snapshots in
  separate workers, so they can move to a dedicated host.
- **Nothing to lock in.** A multi-host profile would be one more `profile=` value,
  and the API itself does not change.
