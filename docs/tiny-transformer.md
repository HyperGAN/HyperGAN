# Tiny transformer generator at 128px

The [recipe](../examples/tiny-transformer-resnet-128.toml) replaces the style
transformer with a [9,851,555-parameter generator](../examples/networks/tiny-transformer-generator-128.hndl).
It keeps the normal-rate recipe's ResNet18 multiscale/pixel critic, no DiffAug,
learned prior, losses, penalties, batch size, seeds, and schedule. Effective
learning rates remain G=0.0002, D=0.0002, and prior=0.002.

The preceding style-transformer runs logged `Nonfinite generator/auxiliary
gradient; run stopped` after completed step 173 (low G rate) and 396 (normal
rate). Those events do not identify the offending operation. This experiment
tests a simpler generator; it is not a diagnosis of those failures.

## Architecture

- One ordinary linear projection maps z=512 directly to 64 tokens of width
  256, arranged as an 8x8 grid. Each position has distinct projection weights.
- Two transformer blocks use LayerNorm before each attention/FFN branch,
  four attention heads, GELU FFNs of width 512, and ordinary residual addition.
- Four nearest-neighbor upsampling stages each double spatial resolution,
  followed by a 3x3 convolution and LeakyReLU(0.2). Output channels are
  128, 64, 32, and 16 at resolutions 16, 32, 64, and 128 respectively.
- A final 3x3 convolution produces RGB, followed by tanh.

All weights use the existing HNDL operator defaults. There is no style mapping,
modulation, zero gate, learned coordinate warp, Fourier encoding, grid sampling,
equalized scaling, or stochastic forward operation. Latents influence the
output at initialization. Existing HNDL operators suffice; this change does
not modify HNDL. The architecture requires a fresh run.

## Validation and launch

The CPU regression check verifies initial image differences between latents,
nonzero latent and transformer/decoder gradients, finite gradients for all
parameters, independence from other batch samples, exact checkpoint roundtrip,
and no forward RNG consumption. Both example and local configurations are
compared with the normal-rate baseline: only generator source and run name
change.

A generator-only float32 batch-64 CUDA forward/backward produced finite
outputs and parameter gradients and a nonzero latent gradient, using 0.817 GiB
peak allocated memory. This excludes the critic, EMA, and optimizer state and
does not establish training stability or convergence.

```bash
../training-runs/start-tiny-transformer-resnet-128.sh
```

The local launcher selects physical GPU 0 and uses a separate run directory:
`/mnt/ml7tb/hypergan-training-runs/train-tiny-transformer-resnet-128`.
Previews remain every 100 steps and checkpoints every 1000. Training is left
for the user to launch.
