# CIFAR with a frozen pretrained critic

The ordinary Python components in `hypergan.image_components` port Martyn
Garcia's ParticleGAN experiment at commit
`9e9ce96c96948197e21e1171c8394e3819bb0013`, branch
`feat/cifar-ae-gan-pretrained-encoder`. The source files are
`experiments/train_cifar_ae_sagan.py`, `lib/image_particle_autoencoder.py`,
`lib/image_moonshots.py`, and `lib/image_ddgan.py`. The recipe follows
`configs/cifar_particle_ae/sagan_gd_16k_200k/sagan_gd_16k.yaml`.
The owner authorized distribution of the port under HyperGAN's MIT license on
2026-09-20 and confirmed that no additional attribution is required: the projects
share authorship. These source references remain for reproducibility; the existing
[MIT license](../LICENSE) applies to the port. This permission covers the source
components, not a redistribution or relicensing of pretrained weights or data.

[The example](../examples/cifar-pretrained-sagan.toml) configures a scratch
encoder, a 16,384-row MoG prior in 64 dimensions, fixed sigma
`0.212616428732872`, and encoder-only reconstruction through a shared generator.
It is an initial HyperGAN image recipe, not a reproduced final FID result.
The source's separate attention-depth intervention is not part of this recipe.

Install `hypergan[train,cifar,web,fid]` from this checkout with a matching CUDA
PyTorch/TorchVision installation.
Set the example's data root and weight path to existing local files before
running it. There are no automatic downloads. The ResNet18 weight file is
`resnet18-f37072fd.pth`, SHA256
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
The constructed frozen feature state must also match
`5de287ab28d569dfc53a5bca4a646d4416621da29e71e80859e6117c7f90b0ac`.
Missing files, changed bytes and mismatched feature states fail explicitly.
Weights and dataset bytes are not included in the package.


## CLI walkthrough

From the repository checkout, install the optional dependencies into your chosen
CUDA environment with `python -m pip install '.[train,cifar,web,fid]'`.
Copy `examples/cifar-pretrained-sagan.toml` to `cifar.toml`. Put the already
obtained official Python-format CIFAR batches under `./data/cifar-10-batches-py/`,
or edit all three `root` entries to their existing location. Paths are relative
to the command's working directory; use absolute paths if you run commands from
multiple locations. The example uses the conventional torch checkpoint cache for
both ResNet18 and Inception; change both Inception entries if your cache differs.
The pinned Inception file is `weights-inception-2015-12-05-6726825d.pth`, SHA256
`6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`.

The following shell session selects physical GPU 1. Choose your device before
training and keep its visibility consistent when resuming. The recipe uses
logical `cuda` within that selection.

```sh
export CUDA_VISIBLE_DEVICES=1
hypergan validate cifar.toml
hypergan preflight cifar.toml --runtime
hypergan train cifar.toml --run-dir runs/cifar-smoke --stop-after-steps 7 --checkpoint-every 7 --preview-every 4
hypergan resume runs/cifar-smoke --stop-after-steps 9
hypergan sample runs/cifar-smoke --count 64 --seed 34002 --output samples.png
hypergan evaluate runs/cifar-smoke --metric fid_smoke
```

This crosses lazy-penalty boundaries and ends at update 16 while preserving the
full 200k training schedule. `fid_smoke` uses only 128 samples and checks the
metric workflow, not image quality. A new longer run uses a separate directory:

```sh
hypergan train cifar.toml --run-dir runs/cifar --stop-after-steps 10000 --checkpoint-every 1000 --preview-every 500
hypergan evaluate runs/cifar --metric fid50k_train
hypergan resume runs/cifar --stop-after-steps 10000 --checkpoint-every 1000 --preview-every 500
```

Evaluation is manual in this example; repeat it after each stopped segment.
`fid50k_train` evaluates EMA against all 50,000 unaugmented training references
with 50,000 generated images, seed 34002 and batch 128. Optional automatic viewer
startup prints its URL and private token-file path; `--no-server` disables it.
For a viewer that persists across stopped segments, run `hypergan serve runs/cifar`
and `hypergan project runs/cifar --follow` in separate terminals. Training prints
per-update D/G losses to stderr; redirect stderr to a log for `tail -F`.

The [execution report](../reports/image-training-execution-2026-09-20.md) records
the exact installed CUDA smoke/recovery proofs and the first 40k measurements.
Keep the installed environment used by a run for supported recovery; source and
package identity are intentionally checked strictly. The public example changes
only artifact locations relative to that measured recipe, not numerical settings.

The generator projects to 256×4×4, then uses three deconvolutions, hidden
GroupNorm/ReLU and a final tanh. SAGAN-style attention appears at 16×16 in
both the generator and the pixel critic. It uses unscaled dot products,
bias-free projections and an active residual of one. There is no learned
gate, spectral normalization or attention warmup. The feature critic combines
pixel scores with frozen ImageNet ResNet18 layer1–layer3 features. Candidate
images are bilinearly resized to 64×64 and normalized with ImageNet mean and
standard deviation, without clamping. Frozen feature parameters and BatchNorm
statistics remain fixed, while input gradients and the penalty's double
backward pass through the feature extractor.

Constructor order is generator, discriminator, encoder. The source consumes
initialization draws for a discarded residual generator, then builds the actual
generator inside a CPU RNG fork. The port preserves that ordering deliberately,
including the private attention seed `124003`. The prior initializes on CPU
with seed `24003`, then transfers to the execution device. These details preserve
the source experiment's initial state; they do not provide historical checkpoint
compatibility. Runtime recovery uses HyperGAN's complete current-run checkpoints.
The critic's constant-context feature cache is derived from frozen weights and
is cleared on device/dtype transfers and state restoration.

Training uses independent real and prior draws for D and G. CIFAR sampling draws
indices with replacement, converts uint8 NCHW bytes using `float()/127.5-1`, and
then draws one horizontal-flip decision per image. Both random operations use
the caller's execution-device generator, in that order. CIFAR Python batch files
are checked against published-byte SHA256 identities before deserialization.
The batch order follows `data_batch_1` through `data_batch_5` without a shuffle
or split. Labels are available but are not inputs to this unconditional recipe.

The reconstruction alias temporarily freezes G's parameters while preserving
the input gradient to E. E detaches the prior means internally. Its hard nearest
particle selection uses a soft straight-through query gradient and a bounded
offset. The ordinary MSE objective contributes only to E; G and the learned prior
still receive their adversarial gradients. The prior regularizer sees all raw
rows. Adam uses fused PyTorch updates, G/E learning rate 0.0003, D 0.00045, prior
0.003, G/D betas `(0, .999)`, prior betas `(.5, .999)`, and constant learning rate.
The bounded candidate penalty is evaluated every eighth update. EMA is `.995`.
The explicit independent-draw/fused policy currently requires native execution;
replicated execution rejects that unsupported configuration.

For FID reference batches, use `hypergan.image_data:CIFAR10Data` with
`split="train"`, `sampling="sequential"` and `horizontal_flip=false`.
Each new evaluator starts at index zero; 50,000 samples consume the full training
set without replacement. Sequential data never wraps. Evaluation has its own
data instance and generator and does not advance training data state. The
standard metrics preset records training scalars; Inception FID is a separately
configured, pinned-weight snapshot evaluator.

Architecture parity compares the pinned source and this port under the same
backend policy. HyperGAN's deterministic execution disables TF32 and cuDNN
benchmarking, while the original source enables them. This is a recorded
execution-policy difference, not a claim of identical trajectories under the
source's original accelerated backend settings. Initial tensor and CPU RNG
comparisons are exact; actual CUDA comparisons declare tolerances before running.

## Deterministic feature execution variant

The example explicitly sets `deterministic_features=true` on the discriminator.
The source CUDA adaptive-average-pool backward uses nondeterministic accumulation;
identical training replays diverged after Adam updates. Strict deterministic
algorithms rejected that backward operation. The failed historical source-weight
comparison remains evidence of a failed gate; its tolerance is not widened.

The deterministic variant keeps all initialized weights, buffers, module state
keys and forward calculations unchanged. Each feature head still calls the source
adaptive-average-pool forward to produce 4×4 output. For the actual 16×16, 8×8
and 4×4 inputs, its backward repeats each output gradient into its nonoverlapping
input cell and divides by the cell area. This analytical adjoint supports higher
derivatives without CUDA atomic accumulation. Bilinear resize retains the native
PyTorch operation: the installed CUDA runtime supports its forward, first backward
and second backward with strict deterministic algorithms enabled.

The example explicitly enables strict deterministic algorithms and deterministic
cuDNN, disables TF32 and cuDNN benchmarking, and sets the cuBLAS workspace to
`:4096:8` before CUDA initialization. This is a declared deterministic execution
variant, not a claim of identical trained weights to the historical
nondeterministic TF32/benchmark run. CPU float64 tests retain exact forward and
first-derivative equality for all three actual feature-map shapes and compare
second derivatives at `1e-14` absolute/relative tolerance. Complete CUDA replay
and current-run checkpoint recovery must be qualified separately under this
strict policy. `deterministic_features=false` retains historical adaptive pooling
for source comparisons and carries no CUDA recovery qualification. Neither
variant changes the pinned pretrained weights.
