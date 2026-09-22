# Early generator saturation in the TransGAN/DINOv3 testbed

The 128px tuned generator has a concrete early failure: by step 20 its online
outputs are almost entirely tanh-saturated, despite a stronger gradient arriving
from the discriminator at the generated pixels. The initial transmission score
passing does not predict preservation of that transmission after updates.

This investigation treats the user's successful ParticleGAN training dynamics
once established as the working premise, and focuses on the startup transition.
It does not propose changing pretrained weights or adding regularization.

## Protocol

A separate run used the user's original config and configured seed, with
`--tune --no-server --checkpoint-every 20 --preview-every 20
--progress-every 20 --stop-after-steps 100`. The complete 200,000-step schedule
was preserved. It ran on physical GPU 1, after verifying no training process
occupied that GPU. The original user run was untouched. Research artifacts live
on `/mnt/ml7tb/hypergan-signal-research/` because the home drive had limited space.

Run: `transgan-init-v2-tuned-100`. Matched probes:
`transgan-init-v2-tuned-100-probes-v2`. The earlier unsuffixed probe directory is
an incomplete attempt that rejected a small TF32 branch-reconstruction error;
it is not the analysis source.

Checkpoints 0, 20 and 100 were evaluated using the same 64 real images, particle
IDs and Gaussian noise. The primary comparison lets learned particle coordinates
evolve. A second comparison keeps the complete initial latent tensors fixed.
The bank may overlap startup calibration/training; this is a matched drift
control, not independent quality evaluation. The scripts take no optimizer steps
and preserve saved runs. Online state hashes match before/after every probe,
and frozen state hashes match across checkpoints.

## Baseline findings

| Online measurement, fixed particle IDs/noise | Step 0 | Step 20 | Step 100 |
| --- | ---: | ---: | ---: |
| Fraction of output values with abs(value) > .99 | 0.0373% | 97.41% | 93.31% |
| Across-sample output standard-deviation RMS | .4553 | .02923 | .06931 |
| Across-sample spread after 4x4 pooling | .04272 | .001968 | .004954 |
| Image gradient RMS | 3.510e-6 | 7.162e-6 | 1.517e-5 |
| First observed G layer / image gradient RMS | .6981 | .03584 | .07179 |
| Pre-tanh RGB activation RMS | .7000 | 9.850 | 19.90 |
| Structural transmission heuristic (smaller is its declared target) | .01168 | .8664 | .7976 |

At step 20 the absolute first-layer gradient is approximately 2.567e-7, compared
with 2.450e-6 at startup. The image cotangent has increased while the first-layer
cotangent has decreased. Signals remain finite and nonzero; this is substantial
attenuation, not proof that every generator parameter has no gradient.

The fixed-latent control has 97.41% saturation and output spread .02926 at
step 20, almost identical to the evolving-prior result. Prior movement alone
therefore does not explain the observed output contraction. Raw prior parameters
move, but the configured prior standardizes its readout centers and uses uniform
sampling probabilities; those probabilities are not learned.

Activation growth is already visible in intermediate residual stages: stage 8
block 1 RMS grows .678 -> 1.921, stage 16 block 1 grows .671 -> 4.842, and the
final stage grows .894 -> 5.276 by step 20. This is not merely an output bias
changing while upstream activations stay fixed.

All 175 pretrained backbone parameter tensors are exactly unchanged through
step 100. G's cumulative parameter displacement is 3.43% of initial parameter
L2 at step 20 and 5.87% at step 100. These aggregate parameter-coordinate
measurements establish that updates occur; they are not per-step optimizer
update ratios or functional sensitivity measurements.

## What the discriminator measurement does and does not establish

The installed logistic Rp generator objective is
`mean(softplus(real_score - fake_score))`, with real scores detached in the
G phase. Its fake-score derivative approaches `-1 / batch_size` as D separates
real from fake. Low D loss therefore does not by itself imply vanishing G
score gradients. D and G training losses also use different phases and draws.

The DINO discriminator has a pixel branch and four feature branches. We measure
branch-specific image-space vector-Jacobian products under the actual loss,
then verify their sum against the total image gradient. Under the configured
TF32 kernels, aggregate relative reconstruction error must be <= .001; the
observed maximum over the six baseline probes is .0008813. Norm shares are not
independent contributions to sample quality. Cancellation ratios have no
universal optimum.

At step 0 the image-gradient norm shares are approximately 5.8% pixel and
7.5%, 18.5%, 38.0%, 30.3% across the four feature heads. At step 20 they are
5.6%, 26.1%, 28.2%, 14.5%, 25.6%. D still transmits an image gradient; this does
not certify its direction is useful or eliminate D's influence on the early
G updates. No branch is automatically rescaled on these measurements.

## Controlled intervention

The follow-up trial changes only G's absolute learning rate from .0002 to
.00002. Because the config expresses other rates as multipliers, it uses
`d_lr_mult = 10` and `prior_lr_mult = 100` to retain D=.0002 and prior=.002.
Initialization, architecture, losses, configured seed, pretrained weights,
and other optimizer settings are preserved. Its config and new run folder are
`transgan-init-v2-g-lr-0p1.toml` and `transgan-init-v2-g-lr-0p1-100` under the
research artifact root. This is a substantive controlled intervention, not a
seed sweep. Loss alone is not the acceptance criterion.

The follow-up completed 100 steps. Step-zero graph/prior, EMA graph/prior,
named RNG streams and global RNG states are exactly equal between trials.
Saved learning rates are G=.0002/D=.0002/prior=.002 in the baseline and
G=.00002/D=.0002/prior=.002 in the intervention. Matched real-image and initial
latent hashes also agree. The existing configured nondeterministic kernels
remain enabled; this is one controlled pair, not a statistical replication.

| Online matched-input measurement | Original G rate, step 20 | 0.1x G rate, step 20 | Original G rate, step 100 | 0.1x G rate, step 100 |
| --- | ---: | ---: | ---: | ---: |
| Output saturation fraction | 97.41% | 13.92% | 93.31% | 26.85% |
| Across-sample standard-deviation RMS | .02923 | .2332 | .06931 | .7789 |
| Across-sample spread after 4x4 pooling | .001968 | .02068 | .004954 | .4519 |
| Image gradient RMS | 7.162e-6 | 1.501e-5 | 1.517e-5 | 1.559e-5 |
| First observed G layer / image gradient RMS | .03584 | .3680 | .07179 | 1.796 |

At step 100 the intervention pre-tanh RMS is 2.238, versus 19.90 in the
original-rate trial. Its fixed-latent control also preserves variation
(spread .7780, saturation 26.91%). This supports a smaller G step as a fix
candidate for the observed early saturation. It is not proof of desirable
sample content, semantic diversity, convergence, or a universal optimum.
Variation can increase without quality improving, and saturated pixel values
can be legitimate for this dataset.

Complete intervention probes are in `transgan-init-v2-g-lr-0p1-100-probes-v2`.
The optional branch attribution on the earlier unsuffixed attempt rejected a
.00117797 relative reconstruction error under TF32, exceeding its declared
.001 limit. That incomplete attempt is excluded from the comparison; the
complete intervention analysis uses the standard G/D boundary diagnostics
without the optional branch decomposition. Baseline branch claims above use
only its successful bounded-error probes. No failed intervention branch value
was accepted by relaxing its limit. The baseline's earlier coordinatewise
acceptance check was replaced by the documented aggregate relative-L2 check
before its complete v2 probe; that change is covered by the analytic tests.

The practical implication for a future automatic tuner is to evaluate the
response to a short sequence of actual optimizer updates, with an independently
checked batch, and include owned optimizer step sizes among the bounded
candidates. That search is not implemented by these research scripts. The
current `--tune` still changes only generator boundary initialization.

## Reproduce and interpret

Research helpers are committed alongside this report:

- `checkpoint_signal_probe.py RUN --steps 0 20 100 --branches --output-dir NEW_DIR`
  produces full matched-input gradient, activation, branch and prior reports.
- `checkpoint_parameter_drift.py RUN NEW_JSON` compares cumulative saved
  parameter displacement without reconstructing models.
- `signal_branch_probe.py` is specific to this named five-branch architecture;
  it is not a universal GAN diagnostic API.

Use the training environment and `PYTHONPATH=src` from the implementation
worktree. Keep the same visible CUDA inventory as the source run (GPU1 UUID
mapped to logical CUDA0). Five analytic CPU branch tests cover decomposition,
anticorrelation, zero/tiny signals and restoration. The matched checkpoint
script also passed six CPU probes across three small saved checkpoints, and its
explicit MoG draw matches native sampling exactly.

The compact machine-readable evidence is
[startup-signal-drift-2026-09-21.json](startup-signal-drift-2026-09-21.json).
Full local reports and immutable checkpoints remain outside git.

Preview metrics use EMA G and EMA prior, 16 samples, and pair-distance RMS;
these online probes use 64 samples and coordinate standard-deviation RMS.
They must not be compared as identical measurements. The working CIFAR recipe
also has a routing encoder and reconstruction objective through a reused
parameter-frozen G. That objective directly trains encoder/prior paths rather
than G's weights, and makes the total objective different from this 128px run.

No current measurement establishes improved independent sample quality or
long-term convergence. The implementation PR remains open and unmerged.
