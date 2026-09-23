# Compaction handoff — 2026-09-22, healthy-control follow-up

## Objective and current conclusion

Discover why logos128 TransGAN collapses while CIFAR32/ResNet works, and establish
a robust fix that preserves healthy models. Finite G/D losses, lower saturation,
or a fraction of initial pixel diversity are not sufficient success criteria.
No robust fix has been established. No normalization change has been made.

This continuation completed five 512-update diagnostic runs, all restored and
finished. No jobs/tools remain pending. Branch is `feat/generator-signal-diagnostic`;
code through `da99d1b6`, with the final reporting/handoff update committed afterward. Commit and
push as you go. PR #382 must remain OPEN, unmerged, automerge disabled; this
explicit instruction overrides generic AGENTS merge guidance.

## New evidence

See [complete report](results/2026-09-22-healthy-control/README.md), raw reports,
CSV measurements and `comparison.svg` in that directory.

| At step 512 | Saturation | Pixel diversity / real | 4x4 pooled diversity / real |
| --- | ---: | ---: | ---: |
| Original CIFAR32 adversarial/ResNet | 0.2233% | 85.9055% | 71.7402% |
| Original logos128/ResNet | 100% | 0.0001745% | approximately zero |
| CIFAR with function-preserving 4096-wide early FFNs | 0.3215% | 100.0115% | 108.0898% |
| Same replication, compensated Adam | 0.1231% | 63.7591% | 51.7267% |
| Unchanged small CIFAR model/recipe on logos32 data | 37.1994% | 109.2010% | 123.2553% |

These are online-G, population RMS spread across a fixed 64-image monitor bank,
relative to real images from each recipe. They are not the previous quantized
preview metric and not independent quality/FID measurements. Cross-recipe banks
and data differ; the three CIFAR runs share exact bank and RNG identities.

**Correction to short-screen interpretation:** original CIFAR briefly reaches
94.09% saturation at step16, then recovers to 0.027% by step64. A 32-step failure
is not proof of persistent collapse. Earlier narrow-FFN and pixelshuffle variants
have only been tested through 32; the original logos/ResNet has now been extended
through 512 and does persistently fail within that window. The earlier architecture
README now explicitly records this qualification.

Raw pixel diversity need not grow monotonically: CIFAR starts at 123% of real pixel
diversity (noise) but only 20% of real 4x4-pooled diversity. At 512 these are86% and 72%.
Pooling reduces fine noise; neither metric proves semantic quality. Do not rank
variants' perceptual quality by these numbers.

## Mechanistic observations and causal width test

The logos generator grows a component shared across images in early FFNs, and
later attention/FFN branches amplify it. At 512 stage16 batch-mean RMS is16.662
with between-image RMS0.590. Pre-tanh mean RMS49.171 with variation0.846 becomes
essentially identical saturated outputs. The shared component can be a spatial
template, not merely a flat color or global scalar mean. Fixed-latent observations
also fail; that excludes prior motion as sole explanation, not all prior effects.

`healthy_control_screen.py` records attention branches, FFNs, residual sums,
upsampling and output. Hooks retain the original batch axis (never folded windows).
Original recipes, seeds, player rates, prior and schedule are unchanged.

`replicated_ffn_screen.py` duplicates each CIFAR early FFN unit: factor4 at 8px,
factor16 at 16px, making all four hidden widths4096. Down columns divide by the
factor, biases stay unchanged. All other tensors, including the entire critic,
copy exactly from the original seeded source. This preserves the initial generator
function (full-network CPU float64 error <6e-15); 243 exact tensors,12 transformed.
It deliberately starts duplicate units; this is NOT independent extra capacity or
an independently initialized wide network. TOML alone does NOT reproduce this
initialization; use the runner.

Adam separately updates each down-weight copy. All three CIFAR runs have first
logged G loss about1.017, but post-update pre-tanh RMS is1.0109/source,
1.7055/replicated,1.0116/compensated. Saturation is0.955%,12.237%,0.959% respectively.
This directly shows why similar loss values miss parameterization-dependent
function response. Nevertheless the uncompensated wider CIFAR model recovers,
so this width alone does not establish the cause of sustained logos collapse.

`--compensated` divides duplicated down-weight LR by factor and up-layer Adam eps
by factor. This restores source functional Adam updates in exact arithmetic while
units stay tied. A float64 regression verifies8 steps and uncompensated divergence.
CUDA/TF32 rounding accumulates into differing later adversarial trajectories;
no bit-identical long-run equivalence or treatment-effect ranking is claimed.
This is a diagnostic control, NOT a validated general scaling rule or Newton step.

Critic interactions remain open: CIFAR has nonzero gradient penalty on 61/64
scheduled steps (max1.506); logos has0/64. The data, critic geometry and trajectory
all differ, so this is not causal evidence by itself. Prior critic swaps exclude a
DINO-specific explanation, not all generator–critic interactions.

## Data-only bridge completed after the user endorsed this direction

[Data bridge report](results/2026-09-22-healthy-control/DATA_BRIDGE.md).
The small CIFAR model on logos32 retains109.20% of real pixel diversity,87.02% of
real spatial diversity and 123.26% of real 4x4-pooled diversity at 512. Saturation is
37.20%, versus 22.59% in real logos. This does NOT prove sample quality or stability
beyond512, but the data change alone did not reproduce large-model diversity
collapse. Pre-tanh RMS is3.007, with batch-mean RMS1.251 and between-image RMS2.735.
The gradient penalty is nonzero on 64/64 scheduled steps, unlike logos128's0/64.

Generator/critic HNDLs are byte-identical to original CIFAR; resolved configs
match except name and training/evaluation data bindings. Initial registered
parameter hash and prior/measurement RNG hashes match CIFAR exactly. Original
seed24002, learned prior/sigma, optimizer, data RNG device and all schedules stay.
`bridge_data:Logos32Data` uses the original pinned manifest404757 training images,
verifies bytes, explicitly decodes32px with Lanczos/white aspect padding, and
preserves CIFAR's sampling with replacement and independent horizontal flips.
Its identity records original 128px manifest policy and effective32px geometry.
The source manifest is unchanged. CIFAR labels are omitted but never consumed by
these unconditional networks. Evaluation bindings also use logos; no FID ran.

Runner: `logos_data_bridge_screen.py`; adapter: `bridge_data.py`; recipe:
`testbeds/cifar-transgan32-logos-data/`. Use the runner or put its directory on
PYTHONPATH for the research factory. New tests cover exact CIFAR-style draws,
normalization/flips, rollback/failure, sequential exhaustion, identity rejection,
and config-only-data scope. This run and all GPU jobs have finished.

## Next causal bridge

Keep logos32 data, critic, latent64 prior, optimizer and resolution fixed. Test
natural generator channel-width scaling:256/64/16 ->1024/256/64, with FFN widths
scaled4x, keeping pixelshuffle, four attention heads and two blocks per stage.
Explicitly copy the source critic/prior initialization so increased generator RNG
consumption does not silently change the critic. This is a proposal, NOT prepared
or run yet. It differs from the earlier deliberate FFN unit replication.
Then separate first bicubic upsampling, later64/128px stages, and critic geometry.
Add critic input-gradient norms and G gradients before/after tanh to distinguish
weak adversarial signal from clipping. A robust fix still needs useful diversity,
quality validation and preservation of healthy models.

The earlier proposed parameter-free per-token channel RMSNorm immediately before
RGB remains UNTESTED. It bounds hidden scale but doesn't remove a shared spatial
template, ensure latent dependence, or prevent RGB weights from rescaling.

## Prior evidence worth retaining

- DCGAN works with DCGAN and freestyle DINO critics. Large logos TransGAN fails
  with DCGAN, freestyle DINO, rebuilt projected DINO, and pretrained ResNet.
- Small adversarial CIFAR ran through 1700 with useful diversity, stopped1825.
  Original encoder/reconstruction CIFAR FID10.93 at 50000 is NOT a FID for the
  adversarial-only recipe.
- CIFAR prior fixes sigma0.212616428732872, NOT particle positions. Positions are
  learned; prior LR0.003. Logos prior LR0.002. Avoid old 'frozen/fixed prior' wording.
- Earlier narrowed FFNs and pixelshuffle screens delay early saturation but don't
  establish whether those variants could recover after32. Four early FFN down
  weights dominate first-step response. The negative-curvature grouped probe
  abstained: never take abs(curvature), floor its rate, or call a manual multiplier
  a Newton step.
- DINO freestyle is not Projected GAN (trained projections, gray context, pixel
  tower, scalar score). Projected testbed uses frozen projections/spatial scores.
- Installed ParticleGAN penalty fix differentiates summed per-image logit means,
  avoiding the64-logit gradient multiplier. PR ParticleGAN#42. This does not change
  spatial adversarial loss. Corrected DINO run still collapsed, stopped703 earlier.

## Validation, artifacts and environment

Ten focused CPU tests passed: `test_joint_rate_probe.py`,
`test_replicated_ffn_screen.py` and `test_logos_data_bridge.py`. All 5 native512-step rollouts restored full trainer
state, preserved source TOMLs and passed before/after/restore protected hashes.
Full-model replication preparations also passed CPU checks in both optimizer modes.
Original artifacts: `/mnt/ml7tb/hypergan-signal-research/healthy-control-v1`.
CSV/SVG exporter: `research/startup_tuning/summarize_healthy_controls.py`.
Plotting-only dependencies are isolated in `/tmp/hypergan-signal-plot-deps`; the
training environment was not modified. No training checkpoints were retained.

- Workspace `/home/martyn/dev/hypergan/generator-signal-diagnostic`.
- Don't overwrite AGENTS.md. No seed sweeps or same-experiment/different-seed tests.
- Python `/home/martyn/dev/hypergan/training-runs/transgan-128-env/bin/python`;
  worktree `PYTHONPATH=src`; never `python -I`.
- Never edit training-runs/logos-* TOMLs. New recipes under testbeds/.
- GPU1 UUID GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce remains reserved. Never launch there.
- All our runs used GPU0 UUID GPU-ed080e41-3193-3755-6756-f3d46c433331. The small
  replication controls shared our GPU0 logos run after checking available memory;
  elapsed times are not hardware benchmarks. All finished; GPU0 is now free.
- Last GPU process query showed only viewer pid71674 on GPU1 (~350MB).
- No subagents requested or used. No normalization/default training changes made.
