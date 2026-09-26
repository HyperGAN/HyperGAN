# Python API proofs of concept: review and recommendation

Branches reviewed: `poc/api-config-first` (38fe27b6), `poc/api-graph` (fabd6550), `poc/api-per-network` (40ab2a2a). All three are pushed. Three judges re-ran the demos on CPU from detached worktrees.

## 1. Recommendation

Build on `poc/api-config-first`. Its model file (`format = "hypergan-model/1"`: `[networks.*]` with explicit `role`, one `[[losses]]` list, and `[data]`, `[train]`, `[samplers]`, `[metrics.custom]`, `[evaluations]`) is already the same file for `hg.load`, for `hypergan train/validate/resume` and for replicated workers. Two judges scored it highest (8, 8), and it was the only POC whose dataset resume survived a comment edit to the user's module.

Take these ideas from `poc/api-graph`:
- Python helpers that accept class and function objects, stored as `module:qualname`, with the importability check that refuses `__main__`, lambdas and local definitions.
- The `[samplers]` engine table with `inputs` bindings, plus `load_inference` and `bundle_state`, so samplers can read any output.
- `resolve_config_at(raw, base)` in place of config-first's heuristic for detecting a Mapping.
- The body-hash guard in `hg.save`.
- Adapters for plain-function losses.
- Keyword names that match the CLI flags (`preview_every`).

Take these ideas from `poc/api-per-network`:
- `explain()` and two-way loss-graph validation, so that explicit roles are checked rather than decorative.
- A reserved per-network `optimizer` slot that raises an explicit engine-limit error until the engine supports it.
- `observations()`, `catalog()`, and `lift()` as a `hypergan convert` from any engine recipe.

Also expose the existing `execution.prepare_train`/`prepare_resume` split as `hg.prepare`/`hg.prepare_resume`, so that `cli.py` becomes argparse plus printing over the API.

## 2. The target model in the recommended API (after grafting)

Encoder, generator, two critics, reconstruction, custom dataset, sampler, metric and evaluation:

```python
# train_demo.py -- user code lives in my_project.py (importable; never copied)
import hypergan.api as hg
from my_project import PairedItems, Encoder, CondGenerator, PairCritic, scatter, g_over_d, radius_gap

model = hg.model(
    name="demo/encoder-two-critics",
    data=hg.dataset(PairedItems, args={"size": 4096, "split": "train"}),   # only __len__/__getitem__
    prior=hg.particles(z_dim=4, count=256),
    networks={
        "encoder":  hg.module(Encoder, role="encoder", args={"code": 4}, inputs={"x": "batch.condition"}),
        "decoder":  hg.module(CondGenerator, role="generator", args={"hidden": 64},
                              inputs={"z": "latent", "code": "encoder.mu"}),
        "pair":     hg.module(PairCritic, role="critic", inputs={"x": "candidate", "condition": "batch.condition"}),
        "marginal": hg.hndl(file="critic.hndl", role="critic", input_shape=["B", 2], output_shape=["B", 1],
                            inputs={"x": "candidate"}),
    },
    losses=[
        hg.adversarial("pair", penalty=1.0),
        hg.adversarial("marginal", weight=0.5, penalty=0.5),
        hg.reconstruction(fn="l1", input="generated", target="batch.real"),
        hg.prior_loss(),
    ],
    samplers={"scatter": hg.sampler(scatter, inputs={"points": "generated", "code": "encoder.mu"}, count=256)},
    metrics={"g_over_d": hg.metric(g_over_d, every=10)},          # parameter names pick update scalars
    evaluations={"radius_gap": hg.evaluation(radius_gap, every=500, samples=512, direction="minimize",
                     data=hg.dataset(PairedItems, args={"size": 512, "split": "holdout"}))},
    train=hg.training(steps=2000, batch_size=64, device="cuda", lr=6e-4),
)
print(hg.explain(model))     # role, optimizer group, and every loss whose gradient reaches each network
hg.save(model, "model.toml") # refuses to overwrite a hand-edited file unless overwrite=True

if __name__ == "__main__":   # spawned workers import this script
    run = hg.train(model, "runs/demo", preview_every=100, checkpoint_every=500)
    hg.metrics(run)["g_over_d"]; hg.evaluations(run); hg.view(hg.sample(run, 256), "scatter", run=run)
```

`args={...}` stays separate from the HyperGAN keywords so that per-network options never collide with constructor arguments. When a class object is passed, `args` is bound against its signature at build time, so the silent `hiden=` typo that graph accepted is caught here.

The saved file is also what the CLI takes:

```toml
# hypergan-model body-sha256 = 3f1c...   (hand edits are fine; hg.save will not overwrite them)
format = "hypergan-model/1"
name = "demo/encoder-two-critics"

[data]
dataset = "my_project:PairedItems"
args = { size = 4096, split = "train" }

[prior]
kind = "particles"
z_dim = 4
num_particles = 256

[networks.encoder]
role = "encoder"
module = "my_project:Encoder"
args = { code = 4 }
inputs = { x = "batch.condition" }

[networks.decoder]
role = "generator"
module = "my_project:CondGenerator"
args = { hidden = 64 }
inputs = { z = "latent", code = "encoder.mu" }

[networks.pair]
role = "critic"
module = "my_project:PairCritic"
inputs = { x = "candidate", condition = "batch.condition" }

[networks.marginal]
role = "critic"
hndl = "critic.hndl"
input_shape = ["B", 2]
output_shape = ["B", 1]
inputs = { x = "candidate" }

[[losses]]
type = "adversarial"
critic = "pair"
penalty = 1.0

[[losses]]
type = "adversarial"
critic = "marginal"
weight = 0.5
penalty = 0.5

[[losses]]
type = "reconstruction"
id = "reconstruction"
fn = "l1"
input = "generated"
target = "batch.real"

[[losses]]
type = "prior"

[train]
steps = 2000
batch_size = 64
device = "cuda"
lr = 0.0006

[samplers.scatter]
fn = "my_project:scatter"
inputs = { points = "generated", code = "encoder.mu" }
count = 256

[metrics.custom.g_over_d]
fn = "my_project:g_over_d"
inputs = { g_loss = "g_loss", d_loss = "d_loss" }
every_steps = 10

[evaluations.radius_gap]
fn = "my_project:radius_gap"
every = 500
samples = 512
batch_size = 64
seed = 7
direction = "minimize"
data = { dataset = "my_project:PairedItems", args = { size = 512, split = "holdout" } }
```

This is equivalent to `hypergan train model.toml --run-dir runs/demo --preview-every 100 --checkpoint-every 500`, with the same fingerprint.

What exists today versus what the grafts add:
- **Already works on the branch:** the file format, lowering, the CLI loading the file, `dataset` / `[metrics.custom]` / `[evaluations]` with `fn`, and bindings such as `encoder.mu`.
- **New from grafting:** `hg.model(...)` as a keyword front end over `from_dict`, class objects in `hg.module`, `hg.sampler` with `inputs`, `explain`, the save guard, and `preview_every=`.

## 3. Question 1: explicit roles vs roles derived from losses

The same small model (an encoder feeding a generator, one critic conditioned on data):

Explicit roles (config-first):
```toml
[networks.encoder]   role = "encoder"    inputs = { x = "batch.condition" }
[networks.generator] role = "generator"  inputs = { z = "latent", code = "encoder" }
[networks.critic]    role = "critic"     inputs = { x = "candidate", condition = "batch.condition" }
[[losses]] type = "adversarial"  critic = "critic"
```

Derived roles (graph):
```python
c, z = hg.data("condition"), hg.latent(4)
enc  = hg.net(Encoder, x=c)
fake = hg.net(Gen, z=z, code=enc)
d    = hg.net(Critic, x=hg.candidate, condition=c)
hg.model(data, [hg.adversarial(d, real=hg.data(), fake=fake)])
# enc is generator-side (upstream of fake); d is a critic (it scores); names/roles come from the list
```

| | Explicit (config-first, per-network) | Derived (graph) |
|---|---|---|
| Reviewability | The role is visible in a diff | The role comes from list position and graph shape; you learn it by reading `describe()` |
| Stability | Reordering losses keeps roles | Swapping two adversarial losses changed which critic became `discriminator`, changed the fingerprint, or raised a naming error (verified) |
| Errors | "critic X is in no adversarial loss", "two generators", role typo (verified) | Silent: an unused Decoder was dropped, and conditioning-only networks were frozen without warning |
| Generality | Config-first rejects two `role="generator"`; a second generator has to be labelled `auxiliary` (verified), which is a lie | Multi-generator and reuse just work (verified: encoder + 2 generators + 2 critics trained) |
| Truthfulness | Config-first ignores `encoder`/`auxiliary` in lowering, so they are decoration | Always matches what trains, since the engine derives roles the same way (`training.py` builds the critic set from adversarial terms; everything else goes to `opt_g`) |
| Hook for per-network settings | A network table is a natural home for `optimizer`, `penalty`, schedule group | No stable handle; graph's kwarg split has no room left |

**Recommendation:** make roles explicit and check them against the derived graph, failing on disagreement.
- Require `role` for generator and critic. Allow several `role = "generator"` networks: the one whose output is `generated` is the engine's main generator, and the others lower as trainable generator-side components.
- Keep `encoder` and `auxiliary` as declared intent that is checked. Warn or fail when an encoder receives no gradient. Refuse or require `trainable = false` when a network only conditions a critic. Fail on a loss that reaches no trainable network (config-first accepts this today).
- Print derived facts (gradient sources, frozen because nothing trains it, which critic is the engine's main one) with `hg.explain` and `hypergan model`, never as silent behavior.
- Longer term, make the engine treat all critics the same, so that "the first adversarial loss is the main pair" stops mattering. It still applies to config-first's lowering today.

## 4. Question 2: losses listed together vs attached per network

The same small model with two critics and a reconstruction:

Listed together (config-first):
```toml
[[losses]] type = "adversarial"     critic = "pair"      penalty = 1.0
[[losses]] type = "adversarial"     critic = "marginal"  weight = 0.5  penalty = 0.5
[[losses]] type = "reconstruction"  id = "reconstruction" fn = "l1" input = "generated" target = "batch.real"
[[losses]] type = "prior"
```

Per network (per-network):
```python
"generator": hg.generator(net, inputs=..., losses=[hg.adversarial("pair"), hg.adversarial("marginal"),
                                                  hg.l1("generated", "batch.real", id="reconstruction")]),
"pair":      hg.critic(net, inputs=..., penalty=hg.k3p(coeff=1.0)),
"marginal":  hg.critic(net, inputs=..., judges=[hg.judge("batch.real", "generated", weight=0.5)],
                       penalty=hg.k3p(coeff=0.5)),
prior=hg.particles(..., losses=[hg.spread()])
```

| | Listed together | Per network |
|---|---|---|
| Matches the engine | 1:1 with `adversarial` + `[[adversarial_terms]]` + `[[objectives]]`; order is explicit and is part of the fingerprint | Order comes from the networks dict: reversing it changed the fingerprint c27256b9 to d52b8283 (verified) |
| Shared terms | Reconstruction trains decoder and encoder and is written once | Needs an arbitrary owner. Attaching it to the encoder was accepted and silently renamed the metric to `loss/objectives/encoder.*`, changing the fingerprint (verified) |
| Adversarial terms | Stated once | Stated twice (`adversarial("marginal")` on G, `judge(weight=0.5)` on the critic), and the weight shows only on the critic |
| Stable ids | One id per term, shared by `override("losses.<id>.weight")`, the metric series and the CLI `--set` | Ids qualified by owner change when the owner changes |
| "What trains this network?" | Not visible without a tool | Visible, but only in the Python source; the saved file keeps the grouping only as comments |

**Recommendation:** list losses together, with user-chosen unique ids. Put per-network settings (optimizer, penalty defaults, schedule group) on the network table. Provide the per-network view as `hg.explain` output, not as the authoring structure.

## 5. Scorecard

| POC | UX lens | Generality lens | Architecture lens | Sum |
|---|---|---|---|---|
| `poc/api-config-first` | **8**: most reviewable file; best errors; growth is additive; Python side is dicts and strings | **6**: roles block multi-generator; no plain-function losses; samplers see only one tensor | **8**: same file for CLI and Python; torch-free lowering; O(1) item resume that survives edits; 52 tests | 22 |
| `poc/api-graph` | **7**: least ceremony for the first model; hidden ordering rule; silent `hiden=` and silent dropped network | **7.5**: most general references, multi-generator, best sampler plumbing; kwarg namespace leaves no room for per-network settings | **6**: writes `<run>.toml` as a side effect on every call; one-way round trip; class-name checkpoint keys; 14 tests | 20.5 |
| `poc/api-per-network` | **6**: best `explain()` and two-way checks; most verbose; double declaration | **7**: per-network optimizer slots; `lift()` both ways; samplers not saved and `output=` unused | **7**: smallest engine change; plain engine TOML with user names; fingerprint depends on dict order; resume blocked by source hash | 20 |

What actually ran, and where judges found claims did not hold:

| POC | Re-verified by judges | Tests | Notes / claims that did not hold or were overstated |
|---|---|---|---|
| config-first | Demo 1 (fingerprint 6beaab2c, d_total 0.6767), demo 2 (c6ba729e, d 1.0000, g 2.7833, recon 1.2963), replicated 2-rank gloo (d_total 0.6545). The CLI `validate` / `train --stop-after-steps` / `resume --config` / `model` on `two_critics.toml` gave losses bit-identical to the Python run. | 51 fast + 1 heavy = 52 passed (verified) | Roles `encoder`/`auxiliary` have no effect. A second generator only trains when labelled `auxiliary`. A plain-function `fn=` objective fails ("missing 1 required positional argument") because it is treated as a constructor. Omitting `type="prior"` sets weight 0, while the raw recipe uses the engine default. `hg.resume` passes `checkpoint=None`. `hg.save` drops comments. A loss that trains nothing is accepted. The `d_over_g` call at step 20 was dropped (reported). |
| graph | `simple.py` (d_total 0.6845), multi-network (992b9818, d 0.9236), replicated (0.6102). CLI train/stop/resume bit-identical to the Python run. | 13 fast + 1 heavy = 14 passed | The `hiden=` typo was accepted and failed later inside a worker. An unused Decoder was silently dropped. Swapping adversarial losses changed the main critic or raised a naming error. `hg.train` writes `<run>.toml` and rejects controls other than `steps` on a built model. Data resume identity includes the source sha256 (read in code, not run). ItemDataset checkpoints are O(N). Some .hndl paths are written absolute. The custom metric published once in 200 steps (reported). |
| per-network | Demo 2 (c27256b9, d 0.9634, g 6.7943, recon 5.2400), `explain()` as reported. CLI train/stop/resume bit-identical. Heavy 2-rank test. | 43 fast + 1 heavy = 44 passed | `sampler(output=)` is never used, and samplers are absent from the saved file. The fingerprint changes with networks dict order. A comment-only edit to the dataset module blocks resume ("data identity ... differs"). A reconstruction attached to the encoder was accepted and its metric renamed. The saved file's per-network grouping is comments only. |

In all three, the full foundation suite shows the same 16 failures as a base export of bff6de3d: `python -I` subprocess tests that cannot import hypergan, plus the wheel-only distribution test. The judges trusted this comparison and did not re-run the full suites. Nothing was run on CUDA. None of the POCs demonstrated LoRA-weight, latent or audio outputs on a trained model; a flat 16-dim latent-shaped output trained in config-first.

## 6. Gaps and engine work on the recommended path

**API layer (branch-local, small)**
- Class and function objects in `hg.module` / `fn=` (graph's import-path check), with `args` bound against the signature at build time.
- A `hg.model(...)` keyword front end, and an optional one-call shorthand for the first model with the same fingerprint as the written form.
- `hg.prepare` / `hg.prepare_resume` wrapping `execution.prepare_train` / `prepare_resume`.
- `checkpoint=` on resume; `previews=` renamed to `preview_every=`; every CLI control passed through.
- `hg.explain`, `hg.describe`, `hg.catalog`, `hg.observations`, `hg.events`, `hg.preflight`, `hg.data_check`, `hg.request_checkpoint`.
- The body-hash save guard, a comment-preserving `--set key=value` that never rewrites the file, and `lift()` as `hypergan convert`.
- Replace the Mapping-detection heuristic in `load_config` with `resolve_config_at(raw, base)`.
- Omitting the prior loss should mean the engine default, or be a hard error. It should not silently mean weight 0.

**Run record**
- Record a name map (file name to engine component) and the model file's `{path, sha256}` in the manifest, without copying the file. Then the viewer, `hypergan model`, metrics and checkpoints show `pair` and `decoder` instead of `discriminator` and `generator`.
- Alternatively, and more invasive: let the engine keep user names as component keys.

**Roles and losses**
- Allow several `role="generator"` networks.
- Validate the loss graph in both directions (per-network's checks).
- Reserve `[networks.X.optimizer]` and raise an explicit engine-limit error until the engine supports it.
- Support plain-function losses through a graph-style adapter.

**Critics**
- Engine: per-term series `loss/adversarial/<id>`.
- Per-critic penalty and optimizer groups.
- Replicated execution support for `adversarial_terms`; today the two-critic model runs single-process only in all three POCs.
- Long term, treat critics symmetrically so the "main pair" disappears.

**Item-level data**
- Keep config-first's `ItemData`: O(1) resume state, nested collation, rollback on a failed read, and an optional `identity()` hook.
- Do not gate resume on the source sha256; record it for provenance only.
- Add prefetch and a worker pool owned by HyperGAN.
- The engine must pass rank and world size to the data factory so each rank loads only its shard (graph's plan/load split). Today every rank loads the whole global batch and slices it.

**Never assume images (the hardest engine item)**
- The engine requires `generated` to be one tensor with the shape of `batch.real` (`training.py`).
- The evaluation worker requires flat tensors with a `real` field.
- Nested dict items and outputs (LoRA-like) fail in all three POCs.
- `real`, `generated` and `candidate` need to become pytrees, with critics taking nested inputs, before the core is type-agnostic.

**Samplers**
- Take graph's `[samplers]` table (`fn`, `args`, `inputs` bindings, `count`, `seed`; not fingerprinted), plus `bundle_state` and `load_inference`.
- Run samplers in the preview worker, and have the viewer render image, audio, points, text and tensor views. Today samplers only apply at read time.

**Metrics**
- An in-process path for cheap plain functions. Today a fresh worker runs per call, and observations are dropped at normal step rates.
- Allow binding objective and per-term values, not only `update.*` scalars.

**Multi-host readiness**
- Keep lowering torch-free with no user imports (already true).
- Write relative .hndl paths only.
- No per-rank file writes, which is the reason to avoid graph's `<run>.toml` side effect.
- All-gather the fingerprint before step 1 and refuse on mismatch.
- User modules must be importable on every host (documented).

**Engine features to add later without API breakage**
- Per-network optimizers, schedules other than `d-then-g-v1`, and several priors: new keys in `[prior]`, `[train]` and `[networks.*]`.

## 7. The CLI on this API

**The config file is the same artifact both ways.**
- `hg.load("model.toml")` and the CLI's `config.load_config("model.toml")` both lower the file through `model_file.lower`. The run records the lowered recipe.
- Verified: the file route, the saved-and-reloaded route, the Python-built route and the CLI route all give c6ba729e. CLI `train --stop-after-steps` followed by `resume --config` was bit-identical to the uninterrupted Python run.
- A model built in Python can be resumed from the CLI in two ways:
  - with no file: `hypergan resume RUN`, which uses the recorded recipe;
  - with a file: `hg.save(model, "model.toml")`, then `hypergan train model.toml --run-dir RUN` or `resume RUN --config model.toml`.
- After the grafts, the manifest also records the file's path, sha256 and the name map.

| CLI command | API call it becomes |
|---|---|
| `new PATH [--device D]` | `hg.save(hg.override(hg.packaged("gaussian-grid"), {"train.device": D}), PATH/"model.toml")` (packaged file = DEFAULT fingerprint, verified) |
| `recipes` | `hg.packaged_list()` |
| `validate PATH` | `m = hg.load(PATH); hg.validate(m)` → prints `hg.resolve(m)` + warnings (torch-free) |
| `preflight CONFIG [--profile P] [--runtime]` | `hg.preflight(hg.load(CONFIG), profile=P, runtime=...)` |
| `data-check CONFIG [--output]` | `hg.data_check(hg.load(CONFIG), output=...)` |
| `train CONFIG --run-dir R [--steps N] [flags]` | `p = hg.prepare(hg.load(CONFIG), R, steps=N, **controls)`; CLI prints `p.warnings` + manual-evaluation hint; `p.run(on_event=output.progress)` |
| `resume R [--checkpoint C] [--config F]` | `hg.prepare_resume(R, checkpoint=C, model=hg.load(F) if F else None, **controls).run(...)` (CLI `resume` has no `--steps`; `prepare_resume` does) |
| `sample R --count --seed --output [--sampler S]` | `hg.sample(R, count, seed, output=)`; with `--sampler`: `hg.view(..., S, run=R)` |
| `evaluate R ID` | `hg.evaluate(R, ID, model=...)` |
| `inspect R` | `hg.open_run(R).manifest` |
| `model R\|FILE [--networks] [--write]` | `hg.describe(R, networks=, write=)` + `hg.explain` section, using the name map |
| `metrics R [--revision]` | `hg.catalog(R, revision)`; series via `hg.metrics(R)` |
| `events R --cursor --limit --max-bytes` | `hg.events(R, cursor, limit, max_bytes)` |
| `checkpoint R [--request-id] [--status] [--attempt-id]` | `hg.request_checkpoint(R, request_id=, attempt_id=)` / `hg.checkpoint_status(R, id)` (moves manifest checks out of cli.py) |
| `serve`, `server-status`, `stop-server`, `project`, `contributions` | Thin re-exports (`hg.serve`, `hg.viewer_status`, ...) over run-directory services |
| new: `convert RECIPE` | `hg.save(hg.lift(load_recipe(RECIPE)), OUT)` |

| Flag | API argument |
|---|---|
| `--steps N` | `steps=N` |
| `--profile P` | `profile=P` |
| `--device D` (`new`), device on train | `override({"train.device": D})`, surfaced as `--set train.device=D` |
| `--preview-every N` / `--no-previews` | `preview_every=N` / `preview_every=0` |
| `--preview-keep`, `--preview-name` | `preview_keep=`, `preview_name=` |
| `--checkpoint-every`, `--max-seconds`, `--stop-after-steps` | same-named keywords |
| service timeout flags | `service_policy={...}` |
| `--progress-every`, `--progress-json` | `on_event=` printer (CLI-side) |
| `--server/--no-server/--port/--server-host/--auth/--public-origin/--open/--dev` | Stay CLI-side as the existing `_training_viewer` context around `p.run()`; optionally `viewer=hg.Viewer(...)` |
| new: `--set k=v` (repeatable) | `hg.override(model, {k: v})`; addresses losses by id or critic name; never rewrites the file |

**What changes in `src/hypergan/cli.py` (426 lines on develop)**
- `_dispatch` branches that call `config.load_config`, `execution.prepare_train`/`prepare_resume`, `artifacts.sample`, `evaluation_cli.run_evaluate`, `model_description`, `metrics.read_catalog`, `run_events.read_event_page` and `run_requests` each become one `hg.*` call.
- The train/resume branch keeps its current shape: prepare, print warnings and hint, then run inside the viewer context. It just calls the API.
- `new` writes a model file.
- Add `--set`, `--sampler` and `convert`.
- Error handling is unchanged, because `ModelFileError` is a `ValueError`.
- What remains CLI-only is argparse, bounded JSON printing, progress output and the viewer lifecycle. That leaves one execution path.

## 8. Decisions for the owner

1. Build on `poc/api-config-first`, with the grafts listed in section 1, and keep `poc/api-graph` and `poc/api-per-network` as reference only.
2. Make the model file (`hypergan-model/1`) the one documented format. The engine recipe TOML would stay readable for old runs and `convert`, but would no longer be documented for authoring.
3. Roles: required for generator and critic, checked against the loss graph, several generators allowed. Encoder and auxiliary declared but checked.
4. Losses: one ordered `[[losses]]` list with user ids. Per-network settings go on `[networks.X]`, and the per-network view comes from `explain`.
5. Network names in runs: a recorded name map now, or the engine keeping user component names. The second is cleaner but changes checkpoint keys.
6. Resume identity: provenance hash recorded but not compared (config-first behavior), confirming that edits to user modules do not block resume.
7. Engine priority order. Proposed:
   1. nested (pytree) `real`/`generated` for non-image outputs;
   2. samplers in the preview worker and viewer;
   3. in-process cheap metrics;
   4. per-rank item loading;
   5. symmetric critics with replicated support;
   6. per-network optimizers.
8. Scope for 2.0.0b1: single-GPU API and CLI on this path with replicated execution kept working, or wait for items 7.1–7.2 first.