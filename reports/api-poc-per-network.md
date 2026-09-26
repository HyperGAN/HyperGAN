# API POC C: networks own their losses (`hypergan.api_per_network`)

Branch `poc/api-per-network` (from develop `bff6de3d`). Code: `src/hypergan/api_per_network/`,
examples: `examples/api/per-network/`, tests: `tests/foundation/test_api_poc_per_network.py`.
Everything below was run on CPU with the owner's `particlegan08-env` (torch 2.14, ParticleGAN 0.8.0,
HNDL 0.8.0) with this worktree's `src` first on the path.

## The API in one page

Plain functions build frozen dataclasses. There is no trainer or GAN object.

| Kind | Functions |
| --- | --- |
| What to build | `net(MyModule, **args)` (your `nn.Module`, referenced as `module:Class`), `hndl(source or file=, input_shape=, output_shape=)` |
| Roles (one per network) | `generator(net, inputs=, losses=, optimizer=)`, `encoder(...)` (any trainable generator-side auxiliary), `critic(net, inputs=, judges=, penalty=, optimizer=)`, `frozen(net, inputs=)`, `shared(name, inputs=, losses=)` |
| Losses (attached to a network) | `adversarial('critic')`, `mse(input, target, weight=, id=)`, `l1(...)`, `loss(my_fn or 'mod:Class', inputs=, weight=, id=)`; the prior takes `spread()` |
| Critic side | `judge(real, fake, weight=, inputs=, penalty=)`, `k3p(coeff=, kappa=, lazy_k=, anchor_weight=, anchor_decay=)` |
| Optimizer (per network) | `adam(lr=, lr_mult=, betas=, **options)` |
| Prior, data | `particles(z_dim, count, losses=, optimizer=)`, `mog(...)`, `gaussian(z_dim)`; `data('gaussian_grid', ...)` (batch-level), `items(MyDataset, **args)` (item-level) |
| Observation | `metric(id, fn, inputs=, every=)`, `evaluation(id, fn, data=, samples=, batch_size=, every=None or N, device=)`, `sampler(name, fn, output=)` |
| Model | `recipe(networks_dict, data=, prior=, training=training(...), observe=[...], sampling={...})` |
| Files | `lower(recipe)` gives the raw config dict; `validate`; `fingerprint`; `save(recipe, path)`; `load(path)` reads any config (including every existing example); `explain(recipe)` |
| Runs | `train(recipe or path or dict, run_dir, **controls)`, `resume`, `run`, `metrics`, `observations`, `evaluations`, `evaluate`, `catalog`, `previews(sampler=)`, `samples(sampler=)` |

The rule of this stance: a network's declaration lists everything that trains it. For a generator
that is `adversarial('discriminator')` plus its objectives. For a critic it is its judges and its
penalty. Its optimizer sits on the same declaration. Lowering enforces it both ways:

- A network whose output a judge scores as fake must list `adversarial(<that critic>)`. A listed
  critic that never judges the network is an error.
- A loss must depend, without detachment, on the network it is attached to. Otherwise lowering
  fails with the networks the loss does reach.

`explain()` also prints the gradient each network receives from upstream losses. An encoder
declares only its own `code_norm`, and `explain` shows that it also receives `generator.reconstruction`
and both adversarial signals.

The configuration is the contract. `lower()` emits only what the declarations state, and the
existing `resolve_config` fills everything else, so `defaults = "particlegan"`, fingerprints,
resume, the viewer and replicated execution are unchanged. `save()` writes a plain HyperGAN TOML
file that `hypergan train FILE` accepts. It is grouped per network, and each `[components.X]` is
followed by the `[[objectives]]` and `[[adversarial_terms]]` that train X. Loss ids are qualified
by their network (`generator.reconstruction`), so the published metric is
`loss/objectives/generator.reconstruction`. User Python is only ever `module:object`. HNDL files
stay file references in the saved config. Nothing is copied into run directories beyond what the
engine already records: the resolved config, which inlines HNDL source.

Engine change (one, in `config.load_config`): it also accepts an in-memory raw mapping, with the
same `file`/`network_files` handling relative to the working directory. That lets `train(recipe, ...)`
run without a temporary file. The run manifest records the resolved config exactly as for a file.

## Demo 1: simplest model (ran)

```python
model = {
    'generator': hg.generator(hg.net(PointGenerator, z_dim=4, width=64),   # your nn.Module
                              inputs={'z': hg.LATENT},
                              losses=[hg.adversarial('discriminator')]),
    'discriminator': hg.critic(hg.hndl(CRITIC, input_shape=['B', 2], output_shape=['B', 1]),
                               inputs={'x': hg.CANDIDATE}),
}
recipe = hg.recipe(model, data=hg.data('gaussian_grid', side=5, noise=0.05),
                   prior=hg.particles(z_dim=4, count=256),
                   training=hg.training(steps=20, batch_size=32, device='cpu', seed=1))
run = hg.train(recipe, 'runs/api-per-network/simple')
hg.metrics(run)['loss/d_total'][-1]
```

Output after 20 CPU steps: `complete: 20 steps`. Last values: `loss/d_total` 0.6380,
`loss/g_total` 0.8578, `loss/gradient_penalty` 0.0039, `loss/prior_regularizer` 0.0978 (20 values each).

## Demo 2: encoder + generator + two critics + reconstruction (ran)

```python
model = {
    'encoder': hg.encoder(hg.net(Encoder, dim=2, code=2), inputs={'condition': 'batch.condition'},
        losses=[hg.loss(code_norm, inputs={'code': 'encoder'}, weight=0.01, id='code_norm')]),
    'generator': hg.generator(
        hg.hndl(CONDITIONAL_GENERATOR, input_shape={'z': ['B', 4], 'code': ['B', 2]}, output_shape=['B', 2]),
        inputs={'z': hg.LATENT, 'code': 'encoder'},
        losses=[hg.adversarial('discriminator'), hg.adversarial('marginal'),
                hg.mse('generated', 'batch.real', weight=1.0, id='reconstruction')],
        optimizer=hg.adam(lr=6e-4, betas=(0.0, 0.999))),
    'discriminator': hg.critic(hg.net(PairCritic, dim=2, condition=2),
        inputs={'x': hg.CANDIDATE, 'condition': 'batch.condition'},
        penalty=hg.k3p(coeff=1.0), optimizer=hg.adam(lr_mult=1.5)),
    'marginal': hg.critic(hg.hndl(file=HERE / 'marginal_critic.hndl', input_shape=['B', 2], output_shape=['B', 1]),
        inputs={'x': hg.CANDIDATE},
        judges=[hg.judge('batch.real', 'generated', weight=0.5)], penalty=hg.k3p(coeff=0.5)),
}
recipe = hg.recipe(model, data=hg.items(PairedPoints, count=1024),
                   prior=hg.particles(z_dim=4, count=64, optimizer=hg.adam(lr_mult=10.0)),
                   training=hg.training(steps=10, batch_size=16, device='cpu', seed=2))
```

It lowers to two objectives, `encoder.code_norm` (the plain function goes through
`adapters:function_loss`) and `generator.reconstruction` (mse). It adds one extra term,
`{"id": "marginal.main", "component": "marginal", "weight": 0.5, "penalty": true, "penalty_coeff": 0.5}`,
and the optimizer section `{"lr": 0.0006, "betas": [0, 0.999], "d_lr_mult": 1.5, "prior_lr_mult": 10.0}`.
It validated, then trained 10 CPU steps. Last values: `loss/d_total` 0.9634, `loss/g_total` 6.7943,
`loss/gradient_penalty` 0.0019, `loss/objectives/generator.reconstruction` 5.2400,
`loss/objectives/encoder.code_norm` 0.0003. `explain()` for the encoder:

```
encoder [encoder] toy_project.nets:Encoder
  optimizer: generator-side Adam, lr = optimizer.lr
  own losses: code_norm
  gradient from: fooling discriminator, fooling marginal, loss encoder.code_norm, loss generator.reconstruction
```

## Demo 3: your own data loader, metric, evaluation and sampler (ran)

```python
class RingPoints:                        # toy_project/data.py: the user writes only this
    def __len__(self): return len(self.points)
    def __getitem__(self, i): return self.points[i]      # tensor -> batch.real; dicts name fields

def g_over_d(g_loss, d_loss): return g_loss / d_loss                    # metric
def radius_gap(batches): ...  # |mean radius(generated) - mean radius(reference)|  (evaluation)
def scatter(samples, *, step=None): ...  # (N,2) -> {'kind': 'points2d', 'ascii': [...], ...} (sampler)

observe = [
    hg.metric('g_over_d', g_over_d, inputs={'g_loss': 'update.g_loss', 'd_loss': 'update.d_loss'}, every=5),
    hg.evaluation('radius_gap', radius_gap, data=hg.items(RingPoints, count=256, seed=1),
                  samples=128, batch_size=32, seed=5, every=10, device='cpu', direction='minimize'),
    hg.sampler('scatter', scatter),
]
recipe = hg.recipe(model, data=hg.items(RingPoints, count=2048, seed=0), observe=observe, ...)
run = hg.train(recipe, run_dir, preview_every=10, checkpoint_every=10)
```

How each piece is wired:

- Data. `items()` lowers to the factory `hypergan.api_per_network.adapters:ItemData` with
  `args.dataset = "toy_project.data:RingPoints"`. HyperGAN owns the epoch permutation, which is drawn
  from the run's data RNG. It also owns the resume position (`state_dict`), provenance
  (`resume_identity` includes the dataset module's sha256), and sharding: replicated execution
  slices the global batch.
- Metric. Lowers to `metrics.custom.g_over_d` (scalar) through `adapters:FunctionMetric`. The user
  source hash goes into the metric description, so an edit changes the definition hash.
- Evaluation. Lowers to a snapshot metric through `adapters:FunctionEvaluation`, using the holdout
  item dataset.
- Sampler. Read-side only; see Gaps.

Output from 20 CPU steps (deterministic; two runs gave identical values):

- `g_over_d`: step 5 = 1.5069. Steps 10, 15 and 20 were dropped with "Previous observation is still
  outstanding".
- `radius_gap`: step 10 = 0.8742. Step 20 was skipped (`worker_busy`).
- The step-10 preview was published and the step-20 preview skipped.
- Fresh samples through the scatter sampler: mean radius 0.137, against about 1.0 for the data. The
  model is 20 steps old, so it is untrained.

The drops and skips are the engine's documented lossy policy. Each observation starts a fresh worker
process, which takes seconds, while CPU steps take milliseconds. The API reports them through
`observations()` instead of hiding them.

## Demo 4: config round trip (ran)

```python
path = hg.save(recipe, 'runs/api-per-network/multi-network.toml')
assert hg.fingerprint(recipe) == hg.fingerprint(path) == hg.fingerprint(hg.load(path))
assert hg.lower(hg.load(path)) == hg.lower(recipe)
```

The fingerprint was `c27256b9607360428c75b4e8eed0ddd2edf86390d0feac072cc1fece73388c81` in all three
cases. `hypergan validate` on the saved file exited 0. `save()` itself re-reads the file and refuses
to return if the fingerprint differs. `load()` lifts any config: every one of the 23 existing
`examples/*.toml` lifts to declarations and lowers back with an identical fingerprint. That includes
color-words, whose reused generator is judged by two critics. It becomes
`shared('generator', losses=[adversarial('discriminator'), adversarial('joint_critic'), ...])`. An
excerpt of the saved file:

```toml
# ---- network encoder: encoder ----
[components.encoder]
factory = "toy_project.nets:Encoder"
args = {dim = 2, code = 2}
inputs = {condition = "batch.condition"}

# loss that trains encoder
[[objectives]]
factory = "hypergan.api_per_network.adapters:function_loss"
inputs = {code = "components.encoder"}
weight = 0.01
args = {function = "toy_project.observe:code_norm"}
id = "encoder.code_norm"

# ---- network generator: generator ----
...
# trained to fool: discriminator, marginal (weights live on those critics)
...
# ---- network marginal: critic ----
[components.marginal.args]
input_shape = ["B", 2]
output_shape = ["B", 1]
file = "marginal_critic.hndl"      # relative to the saved file when they share a tree

# another judgment that trains marginal
[[adversarial_terms]]
id = "marginal.main"
...
```

TOML format: no `tomli_w` is installed, so `toml_io.py` is a 120-line writer. It writes multi-line
HNDL as `'''` literals when that is safe, inline tables for short values, and array-of-tables entries
interleaved per network, which `tomllib` accepts. It never writes `None`: every `None` in a
config is its default. TOML was kept because the existing CLI, `load_config` and HNDL `file`
references already use it.

## Demo 5: reading a run (ran)

```python
run = hg.run(run_dir)                                   # Run(path, run_id, status, steps, config_sha256)
hg.catalog(run)['loss/d_total']['label']                # 'Discriminator total'
hg.metrics(run)                                         # {id: [(step, value), ...]}  (18 ids here)
hg.evaluations(run)['radius_gap']                       # [{'step': 10, 'value': 0.8742, 'status': 'complete', ...}]
hg.evaluate(run, 'radius_gap')                          # on demand: complete, step 20, 0.8626
hg.previews(run)                                        # [Preview(step=10, name='g', shape=(16, 2), value=tensor)]
hg.samples(run, 64, seed=1, sampler=scatter)            # {'count': 64, 'radius': 0.1466, ...}
hg.observations(run)                                    # [(step, id, status, reason)]
```

## Tests

- `tests/foundation/test_api_poc_per_network.py`: 43 pass in about 9 s. The file has 44 tests; the
  44th is `heavy` and runs a two-rank `cpu-replicated-gloo` job from an in-memory recipe with
  item-level data, and it passes as well (`-m ""`: 44 passed, 14.5 s). Coverage:
  - lowering of every role; claim, reach and engine-limit errors
  - save/load fingerprint for all three recipes, and lifting all 23 example configs
  - TOML writer edge cases; `load_config(dict)` giving the same result as the file
  - `ItemData` order and resume
  - training, reading and sampling
  - training from the saved file gives the same run identity as training in memory
  - a run stopped at step 3 and resumed reaches the same final loss as an uninterrupted 6-step run
  - manual custom evaluation
- Fast foundation suite: 775 passed, 16 failed, 12 deselected. The same 16 fail identically on an
  untouched export of `bff6de3d` (16 failed, 68 passed in those files). All come from subprocesses
  started with `python -I`, which cannot import hypergan in this environment, plus one test that
  expects a non-editable install. None are related to this branch.

## Gaps (honest)

1. **Samplers are not in the config.** The engine has no table for a user render function, so
   `sampler()` is read-side (`samples`/`previews`) and is not saved. Proposed engine table:
   `[sampling.samplers.<name>] factory = "mod:fn", binding = "..."`, run by the preview worker.
   Choosing what to generate already lowers: use `sampling.views` or `generated`.
2. **Custom scalar metrics** can bind only the engine's `update.*` scalars (`SCALAR_INPUTS`). They
   cannot bind a per-objective loss such as `loss/objectives/generator.reconstruction`, and they run
   in fresh worker processes, so observations are lossy at high step rates.
3. **Per-network optimizers are declared but only three are executed**: the generator's (`lr`,
   `betas`), the discriminator's (`lr_mult`, guard options) and the prior's (`lr_mult`, `betas`,
   damping). Setting one on any other network raises "Engine limit ... not yet executed". This is
   the slot where separate encoder or second-critic optimizers go once the engine has more than one
   generator-side and one critic optimizer.
4. **Names.** The engine requires the adversarial generator to be named `generator` (its output is
   `generated`) and the first critic `discriminator`, and the first judge of `discriminator` is
   fixed to `batch.real` vs `generated`. Other generators and critics may use any name. Multiple
   generators work as components judged through extra terms.
5. **One K3P configuration** is shared by all critics; only `coeff` may differ per critic or judge
   (validated). Judges inherit their critic's penalty by default. This differs from the engine
   default for extra terms, which is no penalty, and lowering always writes it explicitly.
6. **Redundancy by design.** `adversarial('critic')` on a generator mirrors the critic's judges.
   Lowering checks both sides, and the weight lives on the judge because the engine uses one weight
   for both phases.
7. **Roles are not stored in the engine config.** On `load()` they are inferred: critic = the
   discriminator or any component with extra terms; frozen = `trainable = false`; shared = `reuse`;
   generator = judged as fake; anything else trainable = encoder. Fingerprints are unaffected.
8. **Order.** Lowering groups losses by network. A foreign config whose objectives interleave
   networks (A, B, A) cannot keep its order and warns that the fingerprint will change. None of the
   23 examples do; color-words needed network reordering, which `lift` does.
9. **ItemData** loads synchronously (no worker threads or prefetch yet; `ImageFolder` has both). Items
   must collate to tensors, and `batch.real` must be float.
10. **Output shape.** The adversarial generator's output must be one tensor with the shape of
    `batch.real` (engine). Other outputs, nested or not, are reachable as `components.X.key` bindings,
    views and evaluation `generated`. PNG previews apply only to NCHW images; non-image previews stay
    tensors, as in all demos here.
11. Replicated execution rejects extra adversarial terms, so demo 2 runs only single-process.
12. The resolved config and the run manifest still inline HNDL source. That is existing behavior
    and what makes a run's architecture exact; the saved recipe file keeps the reference.

## How multi-host fits

- The unit of meaning is the config: `lower(recipe)` is deterministic, and user code is only
  `module:object`. Any process can rebuild the model from the config dict plus an importable
  project. Replicated workers already receive only `config_values(...)`, and the heavy test shows
  two ranks rebuilding an in-memory recipe this way.
- For multi-host, each host runs either the same script or `hypergan train model.toml`.
  `train(recipe)` would add one check: all-gather `fingerprint(recipe)` before the first step and
  refuse on mismatch. This is not implemented.
- Data sharding stays inside HyperGAN. Today all ranks draw the same global indices from the shared
  data RNG and slice. `ItemData` already splits `indices(batch, generator)` from `load(indices)`,
  so a host-aware runner can draw globally and load only its own slice. User code does not change.
- Evaluations and metrics already run in separate workers from snapshots. On multi-host they would
  run on one host (rank 0) against the shared snapshot, as they do now.
