# Image FID evaluation

Install `hypergan[all]` or `hypergan[train,image,web,fid]` with matching PyTorch/TorchVision wheels for your CUDA runtime.

`hypergan.image_metrics:InceptionFID` evaluates Fréchet Inception Distance on immutable EMA snapshots against reference datasets. By default, HyperGAN automatically downloads and SHA-256 verifies the official Inception-v3 weights (`weights-inception-2015-12-05-6726825d.pth`) on demand into `~/.hypergan/cache/` with multi-rank file locking. You may optionally supply a custom local weight path via `[evaluations.<id>.args] weights_path = "/path/to/weights.pth"`.

Declare the evaluation metric under `[evaluations.<id>]`. Input bindings default automatically to generated and reference samples. Configure `samples`, `batch_size`, `seed`, `device`, and `[evaluations.<id>.data]`:

```toml
[evaluations.fid50k_train]
factory = "hypergan.image_metrics:InceptionFID"
every = 10000                     # interval in steps; omit for manual-only evaluation
device = "cuda:0"
samples = 50000
batch_size = 128
seed = 10000
on_busy = "skip"

[evaluations.fid50k_train.data]
builtin = "cifar10"
split = "train"
sampling = "sequential"
horizontal_flip = false
```

- **Cadence**: Specify `every = 10000` to evaluate periodically during training. Omit `every` to configure the metric as manual-only (evaluated on demand via `hypergan eval <run> --metric <id>`). (Do not write `mode = "snapshot"` or `trigger = ...`; in `hypergan-model/1` all entries under `[evaluations]` are automatically snapshot evaluations, and `trigger` is inferred from `every`).
- **SciPy 1.18+ Compatibility**: HyperGAN pins `torch-fidelity>=0.4.0,<0.5` alongside `scipy>=1.10`. In torch-fidelity 0.4.0, matrix square roots use `np.linalg.eigvals` instead of scipy's deprecated `sqrtm(..., disp=False)`, fully supporting modern SciPy 1.18+ installations without crashes.

After a bounded training segment has stopped, run a manual metric with:

```sh
hypergan evaluate /path/to/run --metric fid_smoke
hypergan resume /path/to/run
```

The metric ID comes from your `[evaluations.<id>]` configuration. Standalone evaluation is
serialized with training, runs in a fresh supervised process and consumes an
immutable EMA snapshot. Its receipt and independent metric stream record the
snapshot hash, update, reference identity, factory sources and complete settings.
`--bundle` selects an earlier immutable snapshot explicitly. Evaluation settings
can be supplied with `--config` without changing the numerical recipe.

To change the cadence of an interval metric under `[evaluations.<id>]`, set `every`
(or omit `every` for manual-only evaluation):

```toml
[evaluations.fid50k_train]
factory = "hypergan.image_metrics:InceptionFID"
every = 10000
on_busy = "skip"
```

Keep its explicit `evaluation.device`, sample count and reference protocol. The
CIFAR example schedules `fid50k_train` this way. Each accepted interval captures
immutable EMA state at a completed training step, then evaluates asynchronously.
Native snapshot storage also runs in the background; replicated rank-zero file
handoff is synchronous. Snapshot capture/device transfer remains a boundary cost.
One shared evaluator slot bounds resource use; busy intervals are recorded as
skipped, and simultaneous metrics rotate priority. A shared training/evaluation
GPU can contend for memory and compute: a 50,000-sample Inception pass holds its
own weights, activations and reference features on that device while training
continues, so a single-GPU run sees slower steps and higher peak memory around
each interval. `hypergan train`, `resume`, `preflight` and `validate` print a
warning when an interval metric's `evaluation.device` may be the training device;
select an available separate device such as `cuda:1` when one exists. Device
indices refer to `CUDA_VISIBLE_DEVICES`.

When every configured snapshot metric omits `every` (manual-only), the run records
an empty `evaluation_schedule` and no periodic FID is published. That is reported as
a startup notice on stderr, and the viewer's status line for each such metric
reads `Manual`, with no next evaluation step.

Normal completion and step/time budget stops drain accepted evaluations within
their timeout. Signals and training failures cancel and reap the evaluator;
cancellation is distinct from metric failure. Resume starts cadence after the
restored step and retains earlier results under their original attempt identity.
FID is shown only in **Snapshot evaluations**, never in Learning curves. Each
snapshot metric is one tile there, whose status line carries the cadence and the
schedule state even before the first FID value; a metric with no result yet shows
`No evaluations yet · first at step 10000` where its chart will be. Published FID
values are plotted as one chart per metric, with the evaluated source step on the
horizontal axis and one point per evaluation, so a single 50k result is a chart
with one point and later intervals extend the same chart; duration, evaluation
device, busy skips, sample count and protocol stay behind that tile's collapsed
**Details**.
See [local web viewer](local-web.md). Manual `evaluate` remains available for an
interval-configured metric while training is stopped.

The extractor is pinned to torch-fidelity 0.4.0, Inception-v3-compatible 2048
features, float32 with TF32 disabled. Generated and reference RGB tensors are
clamped to `[-1,1]`, mapped with `(x+1)*127.5`, rounded and converted to uint8.
The extractor supplies its TensorFlow-compatible resizing. Float64 centered
moments produce unbiased sample covariance without retaining all features;
torch-fidelity computes the final Fréchet distance. Temporary TF32 settings are
restored even on errors.

A low-count smoke evaluation proves this path executes; it is not a competitive
quality result. The selected ParticleGAN reproduction protocol requires 50,000
generated images and all 50,000 CIFAR-10 training references, with the source
sampling/weight choices. External tables may require a different split or count.
Keep those definitions and metric IDs separate. This evaluator does not establish
historical score reproduction, a leaderboard placement or a quality guarantee.
