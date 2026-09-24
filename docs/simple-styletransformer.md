# Simple StyleTransformer at 128px

The [recipe](../examples/simple-styletransformer-resnet-128-low-g-lr.toml)
replaces the 151,512,455-parameter TransGAN with a 19,036,935-parameter
[HNDL generator](../examples/networks/simple-styletransformer-generator-128.hndl)
and lowers the generator learning rate by ten. It retains the ResNet18
multiscale/pixel critic, disables DiffAug, and preserves the previous prior,
losses, penalties, batch size, seeds, and schedule.

## Architecture

This ports `/mnt/ml7tb/experiments/gemini3/lib/simple_styletransformer_generator.py`
(source SHA256 `a5247b5d23279b65f0660247e6402741dc02e61388166281d5e353311b50e546`).
The requested Fourier renderer and learned coordinate warp are retained:

- Three ordinary linear layers map the existing 512-dimensional latent to
  a 512-dimensional style, with LeakyReLU(0.2). The source default is z=256;
  retaining z=512 preserves the learned-prior configuration.
- A style projection predicts a 2x2 rotation/scale/shear matrix, initialized
  to the identity. There is no translation. HNDL expresses it as `I + delta`.
- Endpoint coordinate grids at 16px and 128px are transformed by that same
  matrix. Concatenating them before Fourier encoding gives both paths exactly
  one fixed frequency table: 192 frequencies drawn from Normal(0,8), followed
  by sine/cosine concatenation, without a 2*pi multiplier.
- Six style-conditioned transformer blocks operate only on the 256 coarse
  tokens, at width 384, with eight attention heads and FFN width 1536.
  The style affine supplies shift, scale, and residual gate for both branches.
  There is no normalization, matching the source despite its `AdaLNBlock` name.
- Bilinear `grid_sample` reads the coarse feature map using the warped fine
  coordinates, with zero padding and `align_corners=False`. This preserves
  the source's convention, including endpoint coordinates and the same warp
  applied to both coarse features and sampling coordinates.
- Concatenated sampled features and fine Fourier features feed an ordinary
  768→384→192→3 pixel MLP, with LeakyReLU(0.2) and final tanh.

The HNDL attention projections are separate Q/K/V matrices. Their Xavier gain
is 1/sqrt(2), matching the variance of PyTorch MultiheadAttention's packed
[3D,D] initialization. Attention biases start at zero. Other ordinary affine
layers use PyTorch defaults. There is no equalized scaling or E2 residual gain.
This is a new architecture and starts a fresh run, without loading TransGAN
weights.

As in the source, all modulation weights/biases and the geometry delta start
at zero. Initial generated images therefore do not depend on z, and the
initial gradient to z/mapping is zero. The geometry and residual-gate rows
receive gradients immediately; a numerical test verifies that one Adam update
at the configured G rate opens nonzero gradients to z and the mapping. This
initial behavior is preserved deliberately, not treated as a successful
convergence result.

## Learning rates

| Parameter group | Previous effective rate | New effective rate |
| --- | ---: | ---: |
| Generator | 0.0002 | 0.00002 |
| Discriminator | 0.0002 | 0.0002 |
| Learned prior | 0.002 | 0.002 |

The configuration uses `lr=0.00002`, `d_lr_mult=10`, and
`prior_lr_mult=100`. Both multipliers are relative to `lr`; raising them
preserves the other groups' rates. The recipe guard checks the effective rates
and equality of every other training setting.

## HNDL support and validation

HNDL adds `broadcast_mul`, `coordinate_grid`, `fourier_features`, and
`grid_sample`. The coordinate grids and Fourier table are persistent buffers,
so checkpoints and EMA copies retain their values. All architecture operations
are visible in the `.hndl` file. The local training environment imports the
editable `../hndl` checkout at [commit 7f7cd3d](https://github.com/HyperGAN/HNDL/commit/7f7cd3d539fe4b238102ab7de2a45c7219e74934)
or later; published HNDL 0.6.0 does not include these operators.

Validation completed:

- 663 HNDL CPU tests passed, including the new operators, shared broadcast
  inference, and the generic operator harness; generated docs and lint passed.
  Reduced-precision CUDA harness cases were skipped.
- 14 HyperGAN tests passed, including the recipe guard and a plain-PyTorch
  reference comparison with active gates and nonidentity geometry. Tests cover
  output and latent/parameter gradients, per-image conditioning, and the
  initial zero gates opening after an update.
- Separately instantiated the supplied Python generator at z=512 and 128px,
  transferred weights, and compared active gates and a nonidentity warp:
  maximum output difference 2.8e-7, latent-gradient difference 7.8e-12.
- A generator-only batch-64 float32 forward/backward on GPU 0 passed with
  finite outputs and gradients, peaking at 14.84 GiB allocated. This excludes
  discriminator activations, EMA, and optimizer state; it is not a full-run
  memory measurement.

PyTorch grid sampling supports the first derivatives needed here but not double
backward through the sampler. The existing critic penalty operates on detached
fake images and therefore does not require double backward through G. CUDA
sampling backward can be nondeterministic, consistent with this recipe's
existing backend setting.

## Local launch

```bash
../training-runs/start-simple-styletransformer-resnet-128-low-g-lr.sh
```

The launcher uses physical GPU 0 and writes to
`/mnt/ml7tb/hypergan-training-runs/train-simple-styletransformer-resnet-128-low-g-lr`.
It keeps batch 64, previews every 100 steps, and checkpoints every 1000 steps.
Training is left for the user to launch; convergence has not been evaluated.
