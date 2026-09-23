# Finite extra-generator warmup

Objective: test whether G needs more updates against each D state at startup,
then return to ordinary fixed-schedule training. No adaptive controller and no
claim that a short screen certifies a Nash equilibrium.

Source: unchanged large logos128 TransGAN and pretrained ResNet18 critic,
seed25002, G/D learning rates2e-4, betas[.5,.999], batch64. The existing matched
512-round baseline is reused, not rerun with a different seed.

## Unchanged-rate 4:1 result

32 rounds of1D:4G, followed by96 rounds of1D:1G. The extra G updates hold prior
parameters and optimizer state fixed; prior still updates once per round.
This totals224 G,128 adversarial D,128 prior updates. Extra updates consume
fresh batches/draws, so subsequent training samples differ from baseline.
Initialization and all measurement-bank/RNG identity hashes match baseline.

| At round128 | Saturated RGB values | Pixel diversity / real | Pre-tanh RMS |
| --- | ---: | ---: | ---: |
| Source1:1 baseline | 99.999873% | 0.004684% | 16.8912 |
| 4:1 warmup at original G rate | 99.999650% | 0.006670% | 29.2759 |
| Same + interpolation penalty-only D steps | 99.992975% | 95.147612% | 59.3472 |
| Same interpolation, G rate/4 during warmup only | 94.662317% | 41.351698% | 12.7248 |
| Same interpolation, retain G rate/4 after warmup | 81.709735% | 46.846280% | 6.3481 |

Extra unchanged-rate G updates did not calm saturation in this128-round window.
By round16 (64 G updates), saturation was99.9853% and diversity0.0567% of real.
It did not recover after switching back to1:1 at round33. This does not establish
permanent failure or rule out all warmup lengths/ratios. No quality/FID measured.
The source baseline at128 has128 G updates; compare both counts when interpreting
the warmup's224 G updates. This is not a compute-matched comparison.

The native capped gradient penalty was zero at every scheduled application.
Full trainer restoration, source preservation and protected-state hashes pass.
Elapsed393.67 seconds including diagnostics. Run finished onGPU0; GPU1 untouched.

## User-directed follow-up

Completed: same4:1 schedule plus an interpolation penalty-only D update before
each extra G step. Both additions end after round32. The penalty encourages
input-gradient norm1 on real/fake interpolates; its direction still comes from
adversarial training. The normal endpoint cap remains unchanged. Separate Adam
state isolates penalty momentum from adversarial momentum. This is a changed
startup objective as well as a schedule change, not merely a repeated old cap.

All96 extra penalties produced nonzero losses and their own optimizer updates.
Native endpoint penalties also became nonzero on all16 scheduled applications.
Initialization and monitoring hashes match the baseline and first warmup run;
the original nondeterministic CUDA/TF32 settings remain, so numerical trajectories
are not promised bit-identical across runs. No different-seed repeats were done.

The interpolation run recovered to52.20%sat/41.96%real pixel diversity at round4,
then grew to98.42%sat/91.07%div at16 and99.85%sat/97.07%div at32. At128 it still
retains95.15%real diversity but is99.993%saturated, with pre-tanh RMS59.35 and
mean tanh derivative0.00002265. Substantial variation is not a quality score:
diverse thresholded patterns can still be completely unusable samples.

The penalty changed the diversity outcome but did not calm saturation. This
is evidence that the critic/training interaction matters; it does not establish
that the original critic alone caused the failure or prove a general fix.
Elapsed416.98s; full restoration/protected/source audits passed. Run finished.

Completed control: same interpolation4:1 warmup with G rate5e-5 during the
first32 rounds only, then original2e-4 and1:1 from round33. This tests whether
smaller steps permit extra G updates to help without excessive activation growth.
D and prior rates stay unchanged; there is no ongoing adaptation.

At round4, the quarter-rate control reached5.40%sat/70.67%real diversity, but
this did not last:89.68%sat/30.89%div at16,70.94%sat/37.48%div at32. After
restoring source G rate and ordinary alternation, round128 is94.66%sat,
41.35%pixel diversity,59.16%spatial diversity and8.71%pooled diversity versus real.
Fixed-latent measurements are similar (94.67%sat/41.26%pixel diversity), so learned
prior motion does not explain away the measured failure. Real saturation is24.06%.
Elapsed416.65s; restoration/source/protected audits pass. This control finished.

Final control completed: keep the selected G rate5e-5 after round32 while
ending extra G/penalty steps as before. The intended contrast is the rate jump at handoff;
the post-warmup schedule is fixed1:1, not an adaptive controller.

The two quarter-rate runs already diverge during their identically configured
warmup (at round8, pixel diversity23.15% versus60.68% of real). Initial parameter
and measurement/RNG hashes match; original nondeterministic CUDA/TF32 execution
is retained. Therefore the final control tests a complete recipe, but comparing
its ending to the previous run is NOT an exact paired estimate of the rate jump's
effect. A causal handoff comparison should fork both continuations from the same
saved warmup state. These disposable runs retain no checkpoints. No seed sweep.

At128, the retained quarter-rate run has81.71%saturation,46.85%pixel diversity,
67.03%spatial diversity and11.72%pooled diversity versus real. Pre-tanh RMS6.35
is lower than in the other candidates, but saturation is still far above real's
24.06%. Fixed-latent results are similar. Elapsed433.19s, all state audits pass.

## Insights and limits

- None of these finite warmups solved the saturation problem in128 rounds.
  Extra unchanged-rate G updates alone still yield nearly identical outputs.
- The interpolation-penalty run preserves substantial diversity while remaining
  almost fully saturated. Critic regularization/training interactions matter;
  it is too strong to say the critic has been completely exonerated.
- Smaller G updates reduce activation amplitude in these observations, but
  transient low saturation at round4 is not sufficient. Retaining a smaller
  fixed rate also does not produce a healthy endpoint in this screen.
- Low G/D losses, high raw diversity, low early saturation, and small updates
  each miss different failure modes. Pixel, spatial and pooled variation need
  to be considered together, and none substitutes for sample-quality validation.
- The interpolation penalty constrains gradient magnitude, not direction toward
  real images or semantic correctness. It does not establish an OCR/text mechanism.
- No test proves Nash equilibrium or long-term failure. Returning to ordinary
  training must be evaluated separately from the warmup transient. A causal
  handoff test needs a shared saved warmup state, not separate nondeterministic
  rollouts with the same seed.

All four runs finished onGPU0, roughly6.6–7.2 minutes each including diagnostics;
GPU1 remained untouched. Twelve focused CPU tests pass (six warmup plus six
existing probe tests). All four restoration/source/protected-state audits pass.
No production --tune default was changed, no ongoing controller was added, and
PR382 remains unmerged. User requested compaction and discussion after completion;
do not launch additional experiments without continuing that discussion.

`outputs.csv` contains both fixed/evolving-prior banks and actual G/D/prior and
penalty-only counts. `comparison.svg` plots saturation, spatial diversity and
pre-tanh amplitude, with the end of warmup marked at round32. All values describe
one64-image matched monitoring bank, not independent FID/quality measurements.

Runner: `research/startup_tuning/generator_warmup_screen.py`; exact semantics in
`testbeds/generator-warmup/README.md`. All raw artifacts for the completed case
are in `logos_g4/`; original output root is
`/mnt/ml7tb/hypergan-signal-research/generator-warmup-v1`.
Production --tune behavior is unchanged. PR382 stays open and unmerged.
