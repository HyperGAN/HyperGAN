# 128px logos with the CIFAR feature-only recipe

The [single-file configuration](../examples/logos-tiny-transformer-resnet-features-128.toml)
embeds both complete G/D HNDL definitions in `args.source`. It adapts the
working [CIFAR experiment](cifar-tiny-transformer.md) to logos while retaining
its latent, prior, optimizer, adversarial objective, and regularization.

## Architecture changes

G keeps z=64, the 8x8 grid of width 256, and two pre-LayerNorm transformer
blocks with four heads and 512-wide FFNs. Its existing 16px/128-channel and
32px/64-channel convolutional upsampling stages are followed by two additional
stages: 64px/32 channels and 128px/16 channels. RGB remains a 3x3 convolution
and tanh. G has 2,511,523 parameters versus 2,489,731 on CIFAR.

As requested, D processes the **full 128px image**, with no spatial resize
before ResNet18. Frozen layer1/2/3 feature maps are therefore 32x32, 16x16,
and 8x8. The three existing feature heads retain their channel widths, 4x4
adaptive pooling, and scalar readouts. D still has 171,779 trainable parameters
and no pixel branch. The fixed context, ImageNet normalization, frozen backbone
and BatchNorm statistics, and score sum divided by sqrt(3) are unchanged.
The backbone checkpoint registers 11,689,512 frozen parameters; these readouts
use only the stages through layer3.

## Recipe and data

The recipe preserves CIFAR's 16,384-particle prior, fixed sigma
0.212616428732872, G/D/prior rates 0.0003/0.00045/0.003, G/D Adam betas
(0.0, 0.999), prior betas (0.5, 0.999), batch 64, seeds 24002/24003,
EMA 0.995, and 200,000-step schedule. Logistic RP and the b_cap penalty
(coefficient 1, kappa 1, lazy interval 8) are unchanged. No encoder,
reconstruction objective, or DiffAug is added. Preview count/seed remain
64/34002, and standard metrics remain every step.

Data uses the existing pinned 128px logo manifest and aspect-fit/white-pad
preprocessing. Unlike CIFAR's replacement sampling and horizontal flips, the
logo loader traverses shuffled epochs without flips. It requires a CPU data
RNG; the prior's RNG policy remains unchanged. The example requires a local
manifest path/hash; the prepared training-runs config supplies both. CIFAR FID
evaluations are omitted because their reference dataset is not this dataset.

The image loader now defaults to four decoder threads and one prefetched batch
of uint8 pixels (3 MiB for this recipe). The existing config needs no edits:
after a graceful stop, rerunning the same launcher resumes with the faster
loader. Both independent real-batch draws per training step remain intact.
See [background loading](image-data.md#background-loading) for tuning and recovery
details. Already-running processes retain the loader code they imported.

## Validation and launch

Five focused CPU tests passed across CIFAR and logo variants, including
initial latent conditioning, checkpoint roundtrip, exact expected input
normalization and native 128px feature shapes, frozen backbone state, and
first/second derivatives. Both example/local configs resolve and were compared
against CIFAR to confirm the settings above. A batch-64 CUDA check on physical
GPU 1 used real logos, pinned ResNet weights, an active b_cap penalty, and one
D and G update with finite gradients. It forced kappa=0 for the active-penalty
check; the training recipe retains kappa=1. Peak allocated memory was 4.687 GiB,
excluding prior and EMA state. This is not a convergence test.

```bash
../training-runs/start-logos-tiny-transformer-resnet-features-128.sh
```

The launcher selects physical GPU 1, leaving the CIFAR run on GPU 0. It uses
the new run directory
`/mnt/ml7tb/hypergan-training-runs/train-logos-tiny-transformer-resnet-features-128`,
with previews every 100 steps and checkpoints every 1000. Training is left for
the user to launch; no existing checkpoint is resumed.
