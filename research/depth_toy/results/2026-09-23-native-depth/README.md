# Depth hurts the native 100gaussians recipe within its 7k budget

The CPU comparison is complete. The original three-layer generator reaches
100/100 modes and 98.94% HQ with EMA. Eight layers reach 85 modes / 49.29% HQ;
sixteen reach 35 modes / 28.49% HQ. Only generator depth changes, with common
weights and critic/prior initialization preserved. The native learned prior
and b-cap remain enabled in all cases. This establishes depth sensitivity in
this recipe, not the cause of the image generator's saturation.

| Hidden layers | EMA modes | EMA HQ | EMA sliced W1 | Seconds, including diagnostics |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 100 | 98.94% | .1582 | 74.5 |
| 8 | 85 | 49.29% | .2252 | 92.1 |
| 16 | 35 | 28.49% | .4363 | 119.8 |

HQ is mass within .09 (3 target standard deviations) of a grid center; a mode
needs >=10 such samples in the 20,000-sample bank. Sliced W1 uses 64 fixed
directions against one shared real bank. Lower is better. These are distribution
diagnostics, not proof of correct within-mode variance. Both online and EMA
results are in [COMPARISON.md](COMPARISON.md); the ranking agrees for both.

The whole comparison took about 4.8 minutes on one CPU thread. The 16-layer
case takes about two minutes, including 20,000-sample evaluations, making this
a practical starting point for controlled interventions. These are observed
wall times, not isolated hardware benchmarks.

## What the signal measurements say

At initialization, between-sample output RMS is .078876 (3 layers), .011058
(8), and .000505 (16). Thus the deepest model already has 156 times less output
variation before any adversarial update. Shared initial layers have identical
activations; the extra chain contracts their signal.

On the first G update, output loss-gradient RMS is similar: .000772, .000756,
and .000861. At the first hidden activation, gradient RMS is .0000698,
.0000112, and .000000956, respectively: 73 times smaller in the deepest case.
The adversarial gradient at the latent inputs similarly falls from .0000636
to .000000857. A learned prior also depends on this derivative through G.
This is an observed initial attenuation, not evidence that the gradient remains
small throughout training: by step128 the deep model's latent-gradient RMS is
.00633, above the shallow model's .00354.

At step1 the deepest critic request is already 99.39% shared energy across
samples, and the realized G update is 98.82% shared. This first update does
**not** demonstrate that G turned differentiated requests into a common one:
the requests already agree, consistent with the tightly contracted outputs.
The shallow control also has 88.29% shared update energy and nevertheless learns.
Its cosine with the request is .646 versus .983 for the deepest case. Neither
large shared fractions nor good first-step alignment alone diagnoses failure.

The toy shows initial forward and backward attenuation plus worse distribution
learning with depth. It does not establish that attenuation alone causes the
final deficit, estimate Jacobian singular values, or validate a correction.
Adam, the moving critic, and subsequent amplification still participate.

## Limits and next useful comparison

- This is a plain LeakyReLU MLP with linear output. The image generator has
  residual/RMSNorm blocks, attention, and tanh. It grows shared activations;
  this toy initially contracts variation. We have a depth-sensitive benchmark,
  not yet a faithful reproduction of the image failure mechanism.
- Added layers also add parameters and change the initial function. Common
  initial tensors do not make the initial generator functions identical.
- Deeper models improve late during annealing. At 7k they underperform the
  shallow model; this is not evidence of irreversible collapse or inability
  to converge with a different budget. There was no extension or tuning sweep.
- A useful next structural control is function-preserving depth insertion,
  for example identity-initialized residual blocks, to separate changed initial
  outputs from the response of a deeper parameterization. A separate bridge
  toward the image generator should add residual normalization and bounded
  output explicitly. Neither follow-up has been run or claimed as a fix.

## Validation and provenance

Five focused tests pass: native training parity, diagnostic non-interference,
matched depth initialization, movement decomposition, and mode-collapse metrics.
The parity test compares exact final G/D/prior and both EMA tensors over four
native updates, including annealing. All three runs completed 7,000 updates
with original seed1234. No seed-only repeats, GPU jobs, source-example edits,
or production training changes. Source hashes were rechecked after completion.

Measured runner commit: `f9eb52ff`. ParticleGAN checkout:
`d162e4c4d4d6ac7b93ef41882f290eb54dee085b`. Full hashes and environment are in
[provenance.json](provenance.json). Configurations and complete numeric reports
are stored under each `depth-*` directory. The compact JSON archive preserves
all recorded values. No training checkpoints were retained.

Original append-only logs and artifacts:
`/mnt/ml7tb/hypergan-signal-research/100gaussians-depth-v1` and
`/mnt/ml7tb/hypergan-signal-research/100gaussians-depth-v1.log`.
Regenerate the comparison with `research/depth_toy/summarize.py REPORT_ROOT`.
