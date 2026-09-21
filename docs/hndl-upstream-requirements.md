# Native HNDL 0.2 integration

HyperGAN's image architecture is defined in `src/hypergan/networks/*.hndl`
and recorded in resolved run configurations. HNDL 0.2.0 supplies the operators
that were missing from the initial 0.1.2 migration. The image integration uses
the released PyPI wheel directly; `image_hndl_ops.py` has been removed. There
are no local implementations of matrix multiplication, constants, channel-bias
addition, or deterministic feature pooling.

## Native operators

| Configuration | Integration |
| --- | --- |
| `matmul(a,b)` | Unfused batched multiplication for explicit SAGAN attention, including second derivatives. |
| `constant(x,D)` | A detached `[B,D]` zero vector with no parameters or random draws, used by the pixel critic's zero-conditioned blocks. |
| `reshape(bias,C,1,1)` then `broadcast_add(features,bias)` | Per-example channel bias with native deterministic broadcast reduction. |
| `adaptive_avg_pool(4)` | Native deterministic feature pooling. Divisible dimensions use reshape and mean; small ragged outputs use explicit windows. Both support higher derivatives. |

Native pooling now supports ragged maps as well as divisible extents. Its
forward agrees with PyTorch adaptive pooling within floating-point tolerance;
its reduction order is not required to match the old forward bit for bit.
The `deterministic_features` constructor argument remains accepted for recipe
compatibility; both settings use the native deterministic 4×4 pooling path.

SAGAN remains explicit convolution, reshape, transpose, matmul, softmax, and
residual-add nodes. Query/key width is `max(1,C//8)` and value width is
`max(1,C//2)`. Scores are unscaled and the residual has fixed unit gain.

## Pretrained features

The ResNet18 stages use native HNDL convolution, batch-normalization, pooling,
and residual-block operators. A Python artifact adapter verifies local weights
and copies checkpoint tensors into those nodes. It does not construct a
separate torchvision network or download artifacts. The original canonical
feature-state digest is preserved, including compatibility with historical
batch-normalization counters.

The colorization integration uses native spectral normalization. DINOv3 still
needs an upstream local-pretrained extension: selectable `forward_features`
with dictionary output `x_norm_patchtokens`, or `get_intermediate_layers` with
`n=[2,5,8,11]`, `reshape=True`, `norm=True` and four outputs from one backbone
pass. The 0.2.0 local provider currently supports the model's ordinary forward
or one captured layer. No local replacement for the missing API is planned.

## Runtime validation

Validation uses `/tmp/hypergan-hndl-venv/bin/python`, with the released HNDL
0.2.0 wheel in that environment rather than an editable upstream checkout.
Native HNDL owns network construction, execution, and copying. HyperGAN does
not patch HNDL runtime internals.

HNDL 0.2.0 currently leaves its runtime dtype checks unchanged after `.double()`.
The existing dtype-conversion regression tests expose that upstream limitation;
no local workaround is included. Float32 image validation is separate from
those failing dtype-conversion cases.

## Image validation evidence

With the released HNDL 0.2.0 wheel, the CPU image, CIFAR, ResNet, data, metrics,
and image-grid selection passed 94 tests. Four existing heavy tests and five
upstream dtype-conversion regression cases were explicitly excluded. Those
five regressions remain in the test suite; they are not marked as expected
failures or silently skipped.

`tests/cuda/test_hndl_images.py` passed all three tests on CUDA device 0 with
`torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`:

- Divisible and ragged native pooling agree with the PyTorch forward within
  float32 tolerance; first and second derivatives repeat bit for bit.
- An actual ParticleGAN b-cap penalty differentiates through the native pixel
  critic, including SAGAN matmul, constant conditioning, and broadcast bias.
  Every trainable parameter receives a finite gradient.

Added CPU contracts verify complete generator replacement through
configuration, exact loading of every pretrained feature tensor, and historical
checkpoint ordering with missing batch-normalization counters.

A separate numerical comparison loaded identical parameters into the configured
networks and the original implementation retrieved from Git history. No legacy
network implementation was added to the working tree. On CPU float32:

| Component | Maximum forward error | Maximum first-derivative error | Maximum second-derivative error |
| --- | ---: | ---: | ---: |
| CIFAR generator | 0 | 0 | Not measured |
| CIFAR pixel critic | 2.9802322387695312e-8 | 4.0745362639427185e-10 | 5.002220859751105e-12 |
| Official pretrained ResNet18 stages 1–3 | 0 | 0 | 0 |

The pixel difference reflects multiplication by a configured reciprocal instead
of Python division by the square root. Parameter initialization order and
checkpoint keys change with the migration; the numerical comparison explicitly
used identical weights and does not assert compatibility with old training
checkpoints or the old random-number consumption order.

The ResNet comparison used the already cached official
`resnet18-f37072fd.pth`, with no downloads. Its artifact SHA256 and canonical
feature-state SHA256 both match the pinned recipe. The latter is
`5de287ab28d569dfc53a5bca4a646d4416621da29e71e80859e6117c7f90b0ac`.
Canonical hashing is necessary because the historical checkpoint stores BN
buffers in a different order from a loaded module state and omits
`num_batches_tracked`; the loader follows PyTorch's zero-counter compatibility
behavior for that older format.
