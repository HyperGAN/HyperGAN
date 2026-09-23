# Logos data in the unchanged working CIFAR32 recipe

Changing only the data bindings does not reproduce logos128's near-total loss of
between-image variation in this 512-update window. At 512 the small model on logos
has 37.1994% saturation, 109.2010% of real pixel diversity, 87.0205% of real
spatial diversity and 123.2553% of real 4x4-pooled diversity. Its pre-tanh RMS is
3.0075, versus 49.178 for the large logos generator. This is evidence about the
failure mechanism, not proof of good logo samples or a robust fix.

| Step | Saturation | Pixel diversity / real | Spatial diversity / real | 4x4 pooled diversity / real | Pre-tanh RMS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.3311% | 80.6986% | 108.3522% | 13.9373% | 1.0636 |
| 1 | 1.2177% | 80.9483% | 108.6207% | 14.7235% | 1.0628 |
| 8 | 5.1732% | 72.9283% | 97.6945% | 19.3642% | 1.3885 |
| 16 | 23.8719% | 118.0845% | 153.5468% | 81.2829% | 2.1541 |
| 32 | 93.8044% | 140.1094% | 128.7750% | 146.7236% | 7.3395 |
| 64 | 52.1901% | 124.3091% | 108.8105% | 127.5053% | 3.9430 |
| 128 | 27.3580% | 100.9542% | 83.7481% | 111.9762% | 2.2851 |
| 256 | 54.7704% | 126.4335% | 145.6342% | 118.6530% | 3.0899 |
| 512 | 37.1994% | 109.2010% | 87.0205% | 123.2553% | 3.0075 |

Real32px logos in this monitor bank have 22.5911%
saturation; real CIFAR had1.413%. Saturation is the fraction of RGB channel values
with absolute value above0.99, not a count of whole images. The logo target itself
contains many extreme values. High saturation with substantial diversity is a
different outcome from the large model's100% saturation and 0.0001745% real diversity.
Nonetheless excessive saturation, noisy texture, or limited semantic coverage can
still make samples poor. No independent quality metric or FID was run.

Spatial diversity subtracts each image's channel means before measuring variation;
4x4 pooling suppresses fine pixel noise. Both help distinguish variation from
pure color changes or fine noise, but neither is a semantic quality test. At 512
mean-color changes explain 64.896% of generated
pixel variance versus 44.720% in real logos.
The fixed-latent control also retains 106.6998% of real
pixel diversity with 33.1019% saturation.

## Exact scope of the data change

- The generator and pretrained ResNet HNDLs are byte-identical to CIFAR source.
  Latent64, stages256/64/16, FFNs1024/256/64, prior and sigma, G/D rates3e-4/4.5e-4,
  betas[0,.999], batch64, original seed24002, EMA and training horizon are unchanged.
- Complete resolved configs match except descriptive name and training/evaluation
  data bindings. The initial registered parameter hash matches the completed CIFAR
  baseline exactly. Prior and measurement RNG hashes also match. The real bank
  changes by design; initial generated output statistics match the CIFAR source.
- The existing verified logos manifest supplies404757 training images. Source
  manifest bytes remain unchanged. Decoder geometry is explicitly32px with
  Lanczos aspect-fit white padding, EXIF correction and white alpha compositing.
  Recovery identity records the original 128px manifest policy and effective32px
  preprocessing separately. Source file hashes are checked per access.
- Sampling remains with replacement, with independent horizontal flips on the
  original execution-device RNG. Float32 normalization on that device remains
  uint8/127.5-1. CIFAR class labels are absent; these unconditional networks do not
  consume labels. Dataset content/count and required32px logo preprocessing change.
- Evaluation data bindings also point to logos, preserving their declarations'
  schedule and avoiding a silently incorrect CIFAR-reference FID. The disposable
  rollout does not execute the FID jobs.

The gradient penalty is nonzero on 64/64 scheduled steps, maximum
2.398769; original logos128 had0/64. This remains an observed difference,
not evidence that changing the penalty would fix 128px training.

## Validation and artifacts

Three new CPU tests check CIFAR-equivalent sampling/flips/normalization, sampler and
RNG rollback including decoding failure, sequential exhaustion, identity mismatch
rejection, and exact configuration scope. The native512-step run passes all
restoration, source-config preservation and protected-state hash audits. It began
at clean da99d1b6 on GPU0; GPU1 was untouched. No checkpoint is retained.

The runner is `research/startup_tuning/logos_data_bridge_screen.py`, the adapter is
`bridge_data.py`, and the recipe is `testbeds/cifar-transgan32-logos-data/`.
Full report, code snapshots and resolved config are in [logos32_data](logos32_data/).
Original artifacts: `/mnt/ml7tb/hypergan-signal-research/healthy-control-v1/logos32_data`.
The shared CSVs and [trajectory plot](comparison.svg) include this fifth run.

## Next controlled change

Keep this32px logos data, critic, latent prior and optimizer fixed, and vary the
generator's channel-width schedule first. A useful next case is quadrupling
channels256/64/16 to 1024/256/64 and their4x FFN widths, retaining pixelshuffle,
32px output, four attention heads and the same block counts. Preserve the source
critic and prior initialization explicitly, since a wider generator otherwise
consumes different initialization draws before the critic. This tests natural
channel/FFN width scaling rather than the deliberately duplicated-unit control.
Bicubic8->16, later64/128px stages and critic-resolution changes can then be
separated instead of changing all of them together. This next case is not run yet.
