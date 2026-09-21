# Native HNDL 0.4.0 integration

HyperGAN's image architecture is defined in `src/hypergan/networks/*.hndl`
and recorded in resolved run configurations. HNDL 0.4.0 supplies the native
operators, dtype tracking, and pretrained readouts required by this migration. The image integration uses
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

The standalone ResNet discriminator example uses the native local `pretrained`
operator with `provider="torchvision_resnet18"`, a SHA256-pinned `.pth`, and
`layer="layer3"`. Its `.hndl` file owns ImageNet normalization, feature selection,
`trainable=False`, and the trainable discriminator head. The host provider only
constructs the external torchvision architecture without downloading weights.

DINOv3 uses native named provider readouts from HNDL 0.2.1. The provider verifies
a clean local source checkout at the configured full Git commit; HNDL verifies
and loads the local SHA256-pinned checkpoint. Configuration selects:

- `readout="patch_tokens"`: `forward_features()["x_norm_patchtokens"]`.
- `readout="multidepth"`: one `get_intermediate_layers(n=(2,5,8,11),
  reshape=True, norm=True)` call, concatenated into one tensor by the provider.
  Native HNDL `split` nodes expose its four maps through named output ports.

The readouts select the math SDPA backend so input gradients support the second
backward required by b-cap. ImageNet normalization, token reshaping, joins,
projections, attention, and discriminator heads remain in `.hndl` files. No
custom HNDL operators or runtime patches are needed.

## Runtime and plan compatibility

Current validation uses the released HNDL 0.4.0 PyPI wheel. Native HNDL owns network
construction, execution, copying, and dtype checks. `.double()` and `.to(dtype=...)`
now retarget floating runtime checks; integer embedding inputs keep their dtype.
Casting a deep copy does not change the original network.

HyperGAN stores HNDL source in resolved recipes and resolves it at construction.
It does not load serialized HNDL plans. An independently saved 0.2.x plan with a
pretrained node must be re-resolved from source with 0.3.0 because pretrained
arguments participate in the semantic digest. Plans without pretrained nodes
are unaffected. HNDL 0.4.0 also loads 0.3.0 plans without re-resolving.
Training state predating the HNDL migration
is rejected by checkpoint contract version 2: parameter names and initialization
order changed, so optimizer state cannot be silently reused.

Alternating discriminator/generator updates preserve each parameter's original
trainability. Frozen pretrained features remain frozen while the discriminator
head trains and gradients flow through the features into generated images.

## Image validation evidence

The HNDL 0.2.1 image tests cover the previously failing dtype conversions,
including divisible/ragged feature pooling and ResNet context-cache invalidation.
Both DCGAN 128×128 batch-64 launch configurations were also checked against the
released wheel for forward execution, generator gradients, and candidate second
derivatives. Frozen ResNet parameters and BatchNorm buffers remain unchanged.

Before the 0.2.1 upgrade, `tests/cuda/test_hndl_images.py` passed all three tests on CUDA device 0 with
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

## One pretrained backbone, multiple feature maps

HNDL 0.3.0 provides native multiple intermediate outputs from one shared
pretrained node. The pixel-plus-multiscale 128px example uses:

```hndl
f1, f2, f3 = pretrained(normalized, ${weights_path},
    provider="torchvision_resnet18", sha256=${weights_sha256},
    layers=("layer1", "layer2", "layer3"), trainable=False)
```

One model executes one forward, stopping after the last selected layer. Each map
keeps its native shape and supports input gradients and second derivatives.
HNDL clones captured hook outputs to preserve values across subsequent in-place
operations. HyperGAN's torchvision provider still uses non-in-place ReLU for
higher-order autograd compatibility. Frozen parameters and BatchNorm state remain
frozen through parent train/eval calls.

`layers=` is mutually exclusive with `layer=` and `readout=` and applies to
provider checkpoints. A single requested layer still returns a one-tuple, which
must be bound explicitly before chaining the next operator. A submodule invoked
more than once before the forward stops is rejected; select its enclosing block.
The three-independent-ResNet workaround is unnecessary and is not used.

## Native batch-axis concatenation and equal splitting

The exact CIFAR-style feature heads concatenate candidate maps with the frozen
features of a fixed zero-image context. HNDL 0.4.0 expresses this in one graph
with one frozen backbone, including normalization and head connections.

The full [128px discriminator](../examples/networks/resnet18-multiscale-discriminator-128.hndl) uses:

```hndl
context = constant(x, 3, 128, 128)
combined = concat(x, context, axis=0)  # [B,3,H,W] + [B,3,H,W] -> [2*B,3,H,W]
# Apply ImageNet normalization to combined, then:
f1, f2, f3 = pretrained(normalized, ${weights_path},
    provider="torchvision_resnet18", sha256=${weights_sha256},
    layers=("layer1", "layer2", "layer3"), trainable=False)
a1, c1 = chunk(f1, 2, dim=0)
a2, c2 = chunk(f2, 2, dim=0)
a3, c3 = chunk(f3, 2, dim=0)
head1_input = concat(a1, c1)  # channel concatenation; batch B restored
```

Intermediate contracts track symbolic `2*B` separately from `B`; `chunk` returns
two equal `[B,...]` outputs with input gradients and second derivatives intact.
Its count is a required positional integer; `dim` is keyword-only and defaults
to 1. Indivisible symbolic shapes fail resolution. A bare `B` cannot be split
because the graph cannot prove it is even; concatenating the two batches first
establishes that proof. External input/output contracts retain batch `B`.

The context is zero in the original `[-1,1]` RGB space (midgray). Concatenate
before ImageNet normalization, so its feature maps represent midgray rather
than a zero tensor in normalized space. Frozen BatchNorm prevents candidate and
context samples from affecting each other's statistics. Context features are
recomputed in the shared forward; no Python cache or network implementation is
needed. There are no outstanding HNDL feature requests for this discriminator.
