# Startup benchmark: establish controls before proposing another calibration

Protocol drafted against `10c051d9`. This document changes no training behavior.
The objective is to compare saturation, preservation of sample variation, and
observed progress on a fixed distribution proxy, including the cost of reaching
that progress. It is not to optimize a combined signal score or to declare a
universal GAN learning rate. No additional experiment or seed sweep was run to
write this protocol.

## Baseline rows and matched controls

First measure the **raw source configuration**, with its G/D rates (currently
2e-4 / 2e-4), no tuner or initialization transform. Compare it with the existing
explicit G=D=1e-4 intervention at the same 32-update horizon. Both retain the
prior rate .002, configured optimizer and annealing horizon, independent phase
draws, native alternating update order, and lazy penalty including its active
coefficient. Use the configured seed 25002; this is one controlled experiment
family, not evidence across seeds.

The existing 1e-4 report is
`/mnt/ml7tb/hypergan-signal-research/joint-rate-1e-4-32-v2.json`, summarized in
[the joint-rate report](../../reports/joint-rate-layer-response-2026-09-22.md). It reaches
97.04% saturation and .01767 between-sample RMS at step32; it is a failed
reference, not an intended target. The tiny-rate selector's older EMA preview
results belong in a historical table, not directly beside matched online-G
statistics. Startup acceptance did not validate that selector's longer run.

Comparison identity must include source/config hashes, effective changes,
initial parameter and protected-state hashes, data/preprocessing identity,
online-versus-EMA choice, seed, batch size, training horizon, device/backend,
observed steps, and metric definitions. Save hashes of actual real-bank tensors,
initial latent values and particle IDs, prior sampling RNG, and measurement RNG.
Matching seed alone does not prove matching inputs. The old joint-rate report
does not contain all these bank hashes: label that identity **unverified** until
it is recovered explicitly; do not silently treat it as a strict matched row.
Do not rerun solely to obtain a more favorable measurement.

Freeze the evaluation bank specification before comparing methods. Prefer
several disjoint fixed banks from an explicitly held-out data split when
available. These are measurement samples, not new training seeds. The current
helper uses one B64 real bank which can overlap training; label it a **monitor
bank**, not independent validation. Report each bank separately before any
summary. Algorithm fitting banks and benchmark evaluation banks must be distinct
for future calibrated candidates; the current bank must not become both the
selection target and the sole success evidence.

## Leaderboard columns, with no composite winner

| Column | Definition and direction | What it cannot establish |
| --- | --- | --- |
| Saturation | Fraction of online RGB values with abs(value) > .99; initial, final, peak, and change in percentage points | Lower is not automatically better: real binary logos can legitimately contain extremes; constant gray scores well |
| Between-sample variation | RMS across-particle standard deviation, plus retention relative to step0 when nonzero | Noise or color-only changes can inflate it |
| Spatial variation | Between-sample variation after removing each image's channel means; also pooled 4x4 variation and color share | Within-image contrast alone does not detect identical samples |
| Feature-distribution proxy | Signed unbiased polynomial MMD² of real/fake frozen DINO final-depth spatial means; final value and Q(0)-Q(t) | Not standard Inception KID, semantic quality, or independent of the critic's feature representation |
| Feature spread | Fake/real feature-spread ratio and retention over time; undefined when denominator is unresolved | Matching spread alone can miss mode loss, memorization, or wrong images |
| Progress onset | First observed sustained proxy improvement, defined below; otherwise “not observed” | Not a population convergence rate or statistically significant improvement |
| Cost | Completed native updates, total elapsed seconds, measurement counts, and calibration overhead if present | Total elapsed with probes is not bare optimizer throughput |
| Audit | Source unchanged, state restored for disposable runs, protected parameters/buffers unchanged | A safety audit is not a learning-quality result |

For tanh outputs also report pre-tanh RMS and derivative statistics to explain
saturation. These are architecture-specific diagnostics, not another ranking
score. Compute the same saturation/variation statistics on the real bank as
context when available. Never replace a missing metric with zero.

The current DINO proxy uses `k(x,y)=(x dot y/384 + 1)^3` and the unbiased MMD²
estimator. Negative finite-sample estimates are valid: preserve their sign.
Use signed differences, not ratios or percentage improvements around zero.
Report bank count and sample count. A single B64 result supplies no precise
uncertainty estimate. Repeating a fixed-state deterministic measurement does
not estimate data sampling uncertainty. If multiple banks or a predeclared
resampling analysis are later available, retain their variation; do not call
an arbitrary floating-point epsilon a statistical confidence bound.

Define the sign-only progress-onset column as the earliest recorded t>0 with
Q(t)<Q(0), followed by Q(u)<Q(0) at **every** later recorded u through the common
horizon, with at least two recorded points from t through the end. Thus a lone
final decrease is not called sustained. With multiple banks, require the
criterion on each bank and expose disagreement. This is a declared descriptive
rule, not a significance test. Include the whole Q trajectory and signed change
so this label cannot hide a tiny noisy improvement or subsequent collapse.
Protocol v2 records synchronized training, diagnostic and preparation time,
plus elapsed time at observations. Onset remains a descriptive observed step;
its timestamp includes the measurement overhead and is not bare training speed.
Algorithm computation and any separately collected probes must also be counted.

Avoid ranking solely by saturation or proxy onset. Present tradeoffs in parallel;
show a Pareto set only if desired, without implying its members are good models.
Evidence of collapse and useful-progress evidence answer different questions.

## Observation horizons and generating-system controls

For startup, use online states at 0, 1, 8, 16, and 32, matching the existing
control and including penalty events. Any longer extension (up to 512 updates in the current helper) must be a
separately declared horizon, never compared as if it were a 32-step result.

For a candidate that merits a longer test after reviewing the baseline table,
predeclare observations at 100, 300, 500, and 512 updates, within the research
harness's 512-update cap. These cross the previous failure window; they are not
a claim that success at512 proves
convergence. This document does not launch or authorize endless successive
candidates. Already collapsed controls need not be extended to prove the same
failure again; display missing later results rather than extrapolate them.

The primary sample distribution is **online G with the current learned prior**,
holding sampled particle IDs/noise fixed across observations. Also evaluate
fixed initial latent tensors to isolate G changes. Report their difference at
the same G; it does not remove the prior's historical effect on training. Keep
EMA results separate and label their lag and sampling prior. For longer-run
quality checks, fixed monitoring IDs should be supplemented by separately
reserved prior draws, using the same evaluation specification across methods.

G/D losses and crossed old/new-player losses are explanatory observations only.
The 1e-4 control improved G loss against both fixed critics while its sample
distribution deteriorated. This rules out treating fixed-opponent descent as
the sole learning-speed metric. Likewise, reduced output motion is not progress.

## Sequence after the baseline

1. Publish the matched-control table and all failed/unresolved rows first.
2. Use read-only FLeRM-style output attribution and GeN model checks to propose
   one concrete change. Separate loss-model validity from useful progress.
3. Check the proposed direction on unused points/banks and one declared coupled
   trial before applying an initialization, player-rate, or layer-rate override.
4. Evaluate that actual intervention under this same benchmark. Record exact
   transformed owned tensors or optimizer groups; protect pretrained state.
5. Only then consider the declared longer horizon and future automation.

The current `--direction` audit adds substantial work: for two banks/player,
16 phase-loss evaluations plus four matching-bank gradients, 16 projection
backwards with eight logical output evaluations, and fixed-input crossed loss
checks. A D loss may include an input-gradient penalty; one logical output
evaluation may execute multiple networks. Keep this explanatory audit out of
baseline learning-time comparisons unless identically enabled, or report its
time separately. The completed joint 1e-4 run took 177.99 seconds **including**
its probes; that cannot be used as a bare 32-update training benchmark.

No arbitrary weighted winner score, LR floor, ideal layer-response target, or
new regularizer is defined here. A candidate can beat a failed control and still
be a poor GAN. The benchmark is intended to make that distinction visible.
