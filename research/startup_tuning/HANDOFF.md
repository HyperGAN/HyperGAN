# Compaction handoff — 2026-09-22

## User intent and next action

The user wants to discover why the logos128 TransGAN collapses while the CIFAR32
TransGAN works well against pretrained ResNet, so the eventual fix generalizes
to future models. Keeping a fraction of initial diversity is not success: assess
how useful diversity develops during training, with real-data references and
quality checks. G/D losses and reduced saturation alone are insufficient.

User requested compaction preparation. No further experiment should start until
they continue. No normalization change has been implemented or tested.

The proposed normalization diagnostic was parameter-free per-token RMSNorm over
channels immediately before the final RGB linear projection:
`h = rms_norm(h, eps=1e-8, affine=False)` on [B, tokens, channels]. It would bound
hidden activation scale; it neither subtracts the across-image common component
nor ensures latent dependence or diversity growth. The RGB weights could also
increase scale again. It is an untested intervention, not an identified robust fix.

On continuation, prioritize instrumenting the working CIFAR32/ResNet recipe with
the same stage measurements before choosing an intervention. Compare common
activation energy versus between-image variation, residual branch contributions,
RGB/pre-tanh scale and latent response, alongside output diversity relative to
real data over time. Locate the earliest difference in healthy versus failing
training. Then make one controlled change at a time to bridge the working recipe
toward logos128, separating data, resolution/depth/width, critic preprocessing,
prior and optimizer differences. Demonstrate both collapse prevention and useful
training, and check the eventual fix against working CIFAR and DCGAN controls.
Do not treat all differences between CIFAR and logos as generator architecture.

## Latest completed experiments

Branch `feat/generator-signal-diagnostic`, code/results through `700c39d3`, pushed.
PR #382 remains open and must NOT be merged, overriding generic AGENTS guidance.

All three 32-update screens use pretrained frozen ResNet18 with source logos128
rates. Same configured seed, monitor bank, and measurement RNG. Variants copy
unchanged tensors exactly from the original initialization; reduced tensors use
leading slices with correct initialization scaling. No seed sweep.

| Case | Saturation at 32 | Initial diversity retained |
| --- | ---: | ---: |
| Original replay with stage instrumentation | 99.653% | 1.031% |
| Four early FFNs 4096 -> 1024 | 93.675% | 10.814% |
| Pixel shuffle starting at 8->16, later widths quartered | 89.050% | 12.483% |

Both variants fail. Pixelshuffle is an independent comparison to original,
not stacked with narrow; it changes channel/FFN widths and attention dimensions
as well as upsampling, so it does not isolate interpolation alone.

Stage evidence: source stage16 RMS grows from 0.671 to 6.497 while across-image
RMS variation changes from 0.640 to 0.571. Pre-tanh RMS reaches 19.226 with 0.744
variation, yet output variation is only 0.00569. Shared activation growth and
subsequent tanh clipping are observed; their upstream cause is still unresolved.
The width changes reduce growth without fixing collapse. No CIFAR stage comparison
yet, and no proof that normalization is missing or is the right fix.

Source replay differs numerically from the previous ResNet screen (97.14%
saturation), despite matching initial parameter and monitor hashes. Configured
nondeterministic CUDA/TF32 remains enabled. Do not overinterpret precise rankings.

Evidence: [architecture results](results/2026-09-22-generator-architecture/README.md).
Runner: `research/startup_tuning/architecture_screen.py` (source/narrow/pixelshuffle).
Optional stage hooks added to `reports/joint_rate_probe.py`. Six probe tests and
ten pretrained-provider tests passed. State restoration, frozen weight integrity,
alignment assertions, config equality outside intended changes all passed.
Original artifacts: `/mnt/ml7tb/hypergan-signal-research/transgan128-ffn-width-v1`.
No checkpoints or long training jobs retained; all our GPU jobs finished.

## Working CIFAR control and an important terminology correction

Recipe: `testbeds/cifar-transgan32-adversarial/cifar-transgan.toml` and HNDLs.
Prior handoff observed saturation <2%, diversity 72–96% of real previews through
step1700, stopped1825. The original encoder/reconstruction recipe achieved FID10.93
at50000; that FID is NOT a measurement of the adversarial-only run.

CIFAR channels256/64/16, FFNs1024/256/64, latent64, pixelshuffle from8px; ResNet
receives bilinear-upsampled64px input. CIFAR uses G3e-4, D4.5e-4, betas[0,.999],
16384 prior particles and fixed_sigma0.212616428732872, versus logos latent128,
4096 particles, G/D2e-4 and betas[.5,.999]. Full TOMLs enumerate further differences.

Earlier prose called the CIFAR prior "fixed". The actual adversarial TOML fixes
sigma, NOT the particle positions: `make_prior` only fills the sigma buffer;
ParticlePrior defaults to learnable z, and the trainer optimizes prior parameters.
Do not carry the frozen-prior claim forward. The CIFAR testbed README also has
stale "not started" wording; prefer recorded run evidence over that sentence.

## Constraints and environment

- Workspace `/home/martyn/dev/hypergan/generator-signal-diagnostic`.
- Commit and push as you go; do not overwrite AGENTS.md; do not merge PR #382.
- Python `/home/martyn/dev/hypergan/training-runs/transgan-128-env/bin/python`.
- Worktree PYTHONPATH; never python -I; no seed experiments.
- Never edit training-runs/logos-* source TOMLs. New recipes under testbeds/.
- GPU1 UUID GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce is reserved; do not launch there.
- Use GPU0 UUID GPU-ed080e41-3193-3755-6756-f3d46c433331 only after checking availability.
- Prior scoremean projectedDINO job was observed stopped703; do not restart implicitly.
- Installed ParticleGAN penalty fix averages logits per image before gradient norm;
  preserves spatial adversarial loss. ParticleGAN PR42. Existing audit evidence in
  results/2026-09-22-projected-scoremean-audit/.
- Earlier grouped curvature probe abstained for negative curvature. Never abs it,
  add a rate floor, or describe the empirical FFN multipliers as a Newton step.
- Prefer metrics; no subagents requested; no jobs or tools pending at this handoff.
