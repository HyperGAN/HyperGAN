# Pretrained ResNet18 does not prevent TransGAN 128 startup collapse

The requested pairing fails the 32-update screen: 97.14% saturated outputs
and only 9.49% of initial between-sample diversity. This is evidence that the
large logos TransGAN's collapse is not specific to the rebuilt DINO critic.
It does not establish a universal failure of TransGAN or pretrained critics.

The [recipe](../../testbeds/transgan-resnet128/README.md) copies the existing
128px multiscale ResNet critic unchanged. Its backbone is frozen ImageNet
ResNet18, loaded from local `resnet18-f37072fd.pth`, SHA256
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
The trainable heads read layer1/2/3 at 32/16/8px, with a pixel tower and fixed
gray context. This is the existing CIFAR-family critic adapted to 128px, not
a fresh critic design. This changes the whole discriminator relative to the
projected DINO control, not only its backbone.

## Controlled recipe and screen

Compared with the projected-DINO control, resolved configurations are exactly
equal after removing the recipe name and discriminator component. Generator
source bytes are identical. Original G/D rates are 0.0002, prior rate 0.002,
Adam betas [0.5, 0.999], batch 64, seed 25002, full horizon 200000, and lazy
b-cap every eight updates. No FFN multiplier, startup tuning, or seed change.

The declared [screen](../../configs/transgan-128-resnet-screen.json) ran 32
disposable updates on idle GPU 0 (`GPU-ed080e41-3193-3755-6756-f3d46c433331`),
from clean commit `b0627d6d`, in 98.86 seconds including diagnostics. The
DINO-specific feature observer is disabled. There is no retained training
checkpoint or independent quality estimate.

| Update | Saturation | Between-sample diversity RMS | Pre-tanh RMS | Mean tanh derivative |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.010% | 0.551861 | 0.999 | 0.61251 |
| 1 | 26.310% | 0.333918 | 2.242 | 0.29194 |
| 8 | 87.716% | 0.080871 | 7.216 | 0.02865 |
| 16 | 96.219% | 0.056091 | 9.859 | 0.01216 |
| 32 | 97.140% | 0.052399 | 12.863 | 0.00905 |

These are unquantized online-G measurements on 64 fixed particle IDs/noise
with the evolving prior. Diversity is population spread, not preview pairwise
RMS. The initial output measurements and monitor-bank hash exactly match the
earlier projected-DINO control. That control ended at 97.825% saturation and
6.93% diversity retention; neither result warrants promotion to a successful
candidate. Configured nondeterministic/TF32 kernels remain enabled.

The fixed-initial-latent control ends at 97.141% saturation and 0.052426
diversity RMS, nearly identical to the evolving-prior result. This isolates
little instantaneous prior effect at final G, not all prior effects on training.
Pre-tanh activations grow while sample variation contracts; by step 32 the
mean tanh derivative has fallen about 68-fold. Final G loss is 6.7862 and
D adversarial loss 0.001511. The b-cap penalty is zero at each of the four
scheduled applications; the cap is not exceeded there. Saturation and diversity,
rather than loss values, establish failure.

## Integrity and artifacts

Protected pretrained/frozen parameter and buffer hashes match before/after
training and restoration:
`d77cb8fe8cf6e434dc384f5a7efc0546b028bbb77d587159cff710a5594cdb0f`.
Full trainer-state restoration and unchanged source-config audits passed.
The 10 existing pretrained-provider tests passed, including frozen eval mode,
input first/second derivatives, normalization, shared backbone readouts, and
trainable-head updates. The local weight file checksum was independently
verified. Launcher syntax and resolved config equality checks passed.

Complete evidence: [report](source/report.json), [request](source/request.json),
[resolved recipe](source/resolved-training-config.json). Originals are under
`/mnt/ml7tb/hypergan-signal-research/transgan128-pretrained-resnet-v1/source`.

The ordinary launcher is available at
`~/dev/hypergan/training-runs/start-transgan-resnet-multiscale-128.sh`, using
GPU 0 and a separate fresh run directory. It was not started because this
bounded screen already reproduces collapse. No training-runs source TOMLs
were edited; GPU 1 was untouched; PR #382 remains unmerged.
