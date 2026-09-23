# Comparison with the official TransGAN implementation

Reviewed [VITA-Group/TransGAN](https://github.com/VITA-Group/TransGAN/tree/6b85440ca56716fd7a60bac964466cc0296ce663)
at commit `6b85440ca56716fd7a60bac964466cc0296ce663` on 2026-09-23.
The comparison target is our
[`transgan-projected-dinov3-128-stable.toml`](../examples/transgan-projected-dinov3-128-stable.toml).

The official checkout has no dedicated 128px launcher. Its released high-resolution
launchers use 256px variants, with additional mechanisms beyond paper section 3.4.
Our 128px spatial/channel schedule follows [paper Table 6](https://arxiv.org/html/2102.07074v4#A2.T6).
Different 256px widths are therefore not evidence that our 128px implementation
is wrong. This comparison does not establish a cause of training collapse.

## Generator

| Item | Our 128px generator | Paper / official implementation |
| --- | --- | --- |
| Channel schedule | 1024, 1024, 256, 64, 16 at 8, 16, 32, 64, 128px | Matches paper 128px schedule |
| Upsampling | Bicubic 8→16, then PixelShuffle | Matches paper 128px schedule |
| Block depths | 2, 2, 2, 2, 2 | Paper 128px: 5, 4, 4, 4, 4 |
| Latent width | 128 | Paper and released high-resolution scripts: 512 |
| Normalization | Channel RMS, epsilon 1e-8, no affine parameters | Matches PixelNorm |
| Position encoding | Absolute embeddings plus relative attention bias | Matches |
| Attention / MLP | Four heads, no QKV biases, output bias, GELU, MLP ratio 4 | Matches baseline defaults |
| Output | Tokenwise RGB linear followed by **tanh** | RGB convolution with **no tanh** |
| Initialization | Fan-in defaults for linear layers, Xavier RGB weights, truncated-normal position tables | Matches effective launcher initialization |
| Extra randomness | None beyond supplied latent | Selected high-resolution variants add learned-strength attention noise |

The baseline's [PixelNorm and attention](https://github.com/VITA-Group/TransGAN/blob/6b85440ca56716fd7a60bac964466cc0296ce663/models_search/ViT_custom_local544444_256_rp.py#L22-L154)
match our arithmetic. Its [RGB return](https://github.com/VITA-Group/TransGAN/blob/6b85440ca56716fd7a60bac964466cc0296ce663/models_search/ViT_custom_local544444_256_rp.py#L433-L445)
has no bounding activation. Our final `tanh` can attenuate generator gradients
when RGB preactivations grow; given observed saturation, this is a concrete
difference to investigate. It remains in this augmentation-focused configuration.

The [upstream initializer](https://github.com/VITA-Group/TransGAN/blob/6b85440ca56716fd7a60bac964466cc0296ce663/train_derived.py#L83-L105)
only overrides convolution weights; the linear override is commented out.
Our tokenwise RGB projection is the convolution-equivalent exception. Separate
Q/K/V modules have the same fan-in distribution but a different random draw layout.

The released CelebA256 model additionally uses
[latent cross-attention](https://github.com/VITA-Group/TransGAN/blob/6b85440ca56716fd7a60bac964466cc0296ce663/models_search/Celeba256_gen.py#L191-L218)
and [up/down filtering after its 64px and 128px stages](https://github.com/VITA-Group/TransGAN/blob/6b85440ca56716fd7a60bac964466cc0296ce663/models_search/Celeba256_gen.py#L491-L513).
The noisy variants inject [learned-strength token noise](https://github.com/VITA-Group/TransGAN/blob/6b85440ca56716fd7a60bac964466cc0296ce663/models_search/ViT_custom_local544444_256_rp_noise.py#L100-L107),
initialized at zero. These are absent from our reduced-depth generator.

## Augmentation

The initial implementation used generic DiffAug defaults. The final stable recipe
instead adopts the official repository's basic three-operation policy:

| Setting | Generic DiffAug defaults | Final stable recipe / official basic policy |
| --- | --- | --- |
| Order | Color, translation, cutout | Translation, cutout, color |
| Translation maximum | 12.5% per axis | 20% per axis (26px at 128px) |
| Cutout application | Every batch | 30% of batches |
| Cutout region | Half image height and width, clipped at boundaries | Same |

Sources: [official transform arithmetic](https://github.com/VITA-Group/TransGAN/blob/6b85440ca56716fd7a60bac964466cc0296ce663/models_search/diff_aug.py#L271-L362)
and [basic policy selection in the CIFAR script](https://github.com/VITA-Group/TransGAN/blob/6b85440ca56716fd7a60bac964466cc0296ce663/exps/cifar_train.py#L57).
Color runs after spatial transforms, so it also transforms the padded/erased
regions. Our batchwise cutout gate uses device PyTorch RNG rather than Python RNG
to preserve checkpoint replay and avoid a GPU-to-host synchronization. Random
streams are consequently not bitwise equivalent to upstream.

Placement agrees: augmentation is inside the discriminator before feature
extraction, including generator-loss and gradient-penalty calls. We deliberately
disable augmentation in evaluation; upstream uses explicit `aug` arguments
rather than PyTorch training mode. The published
[Church256](https://github.com/VITA-Group/TransGAN/blob/6b85440ca56716fd7a60bac964466cc0296ce663/exps/church_256_train.py#L58)
and [CelebA256](https://github.com/VITA-Group/TransGAN/blob/6b85440ca56716fd7a60bac964466cc0296ce663/exps/celeba_hq_256_train.py#L58)
scripts use stronger dataset-specific policies, including ratio erasing and,
for CelebA, filtering and hue. Those are outside the requested three operators.

## Training recipe

The [released CelebA256 launcher](https://github.com/VITA-Group/TransGAN/blob/6b85440ca56716fd7a60bac964466cc0296ce663/exps/celeba_hq_256_train.py#L16-L60)
and [training loop](https://github.com/VITA-Group/TransGAN/blob/6b85440ca56716fd7a60bac964466cc0296ce663/functions.py#L128-L217)
also differ materially from our projected-DINO setup:

| Item | Our stable recipe | Released high-resolution recipe |
| --- | --- | --- |
| Critic | Frozen DINOv3, fixed projections, learned spectral conv heads | Learned multi-scale transformer discriminator |
| Objective | Relativistic logistic | WGAN-GP plus score-drift term |
| Gradient penalty | Endpoint b-cap, coefficient 1, every eighth step with lazy scaling | Interpolated GP targeting norm 1, coefficient 10 |
| Adam | LR 0.0002, betas (0.5, 0.999) | LR 0.0001, betas (0, 0.99) |
| Update schedule | One D update then one G update | Four D updates per G update |
| G:D batch ratio | 1:1 (64 each) | 2:1, with distributed accumulation |
| Gradient clipping | No global gradient norm clipping | Norm 5 for G and D |
| Prior | Learned 4096-particle MoG, latent width 128 | Standard Gaussian, latent width 512 |
| EMA | 0.995 | 0.995 in the high-resolution launchers |

Normalization and positional bias are verified, not missing fixes. The new
augmentation closes one gap, while output saturation, capacity, optimizer
dynamics, prior and discriminator feedback remain independent differences.
No training-quality or convergence claim follows from this code comparison.
