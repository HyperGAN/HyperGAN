# Explicit image FID

Install `hypergan[train,cifar,fid]` with matching PyTorch/TorchVision wheels for
your CUDA runtime. `hypergan.image_metrics:InceptionFID` is an ordinary Python
snapshot metric. It never downloads weights: provide the local
`weights-inception-2015-12-05-6726825d.pth` file, whose SHA256 must be
`6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`.
Missing or incorrect files fail metric preflight before training starts.

Bind `generated = "evaluation.generated"` and
`reference = "evaluation.reference"`, use `mode = "snapshot"` and
`trigger = "manual"`, and pass the local weight file in `args.weights_path`.
The evaluation configuration must explicitly name the data factory and its
arguments, sample count, batch size, seed and device. The data factory owns
reference ordering and preprocessing; use unaugmented references for CIFAR FID.

After a bounded training segment has stopped, run:

```sh
hypergan evaluate /path/to/run --metric fid_smoke
hypergan resume /path/to/run
```

The metric ID comes from your `metrics.custom` configuration. Evaluation is
serialized with training, runs in a fresh supervised process and consumes an
immutable EMA snapshot. Its receipt and independent metric stream record the
snapshot hash, update, reference identity, factory sources and complete settings.
`--bundle` selects an earlier immutable snapshot explicitly. Evaluation settings
can be supplied with `--config` without changing the numerical recipe.

The extractor is pinned to torch-fidelity 0.3.0, Inception-v3-compatible 2048
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
