# First standardized baseline: neither rate pair learns successfully

Both declared controls fail the 32-update startup screen. This establishes a
baseline for comparing tuning solutions; no new algorithm is promoted.

| Solution | G / D LR | Saturation at 32 | Output diversity retained | DINO MMD² initial → final | Sustained proxy improvement |
| --- | --- | --- | --- | --- | --- |
| Source configuration | .0002 / .0002 | 99.47% | 1.46% | .33885 → .48586 | Not observed |
| Half rates | .0001 / .0001 | 98.96% | 6.47% | .33885 → .36123 | Not observed |

See [the generated leaderboard](leaderboard.md) for absolute values, budgets,
timing and comparison identity; [leaderboard.json](leaderboard.json) retains
the trajectories. Each solution directory contains its original raw report,
request, tuning solution, original training TOML and resolved training JSON.
Relative to the common .55186 initial diversity RMS, final diversity is .00807
and .03571. The real monitor bank has 24.06% saturated values, so saturation
alone is not a universal failure criterion; here it accompanies severe loss
of between-sample variation and no sustained feature-distance improvement.

The half-rate control has better final feature distance and greater feature
spread than the source control, but it still shows strong saturation and sample
variation loss. No winner or aggregate score is assigned. The DINO metric is a
signed unbiased polynomial MMD² estimate on one B64 monitor bank, using a
representation that also participates in the critic. It is not independent
semantic-quality validation or a precise population estimate.

## Reproduction and audit

Measured revision: `74475eded4f7d82f5cdd53d1402cbbb68f23b844`, clean worktree,
after merging develop into this branch. GPU 1 was selected by UUID; GPU 0 was
untouched. Both use seed 25002, matching source-config fingerprints, initial
parameter and evaluation-bank/latent/RNG hashes, prior rate .002, training
horizon 200000, B64 and native alternating/lazy-penalty schedules. No
initialization scales or layer multipliers were applied to either control.

The manifest is [transgan-128-screen.json](../../configs/transgan-128-screen.json).
Tuning solutions are [source.json](../../configs/solutions/source.json) and
[half-rates.json](../../configs/solutions/half-rates.json). The invocation is in
[the benchmark guide](../../README.md). Original artifacts remain at
`/mnt/ml7tb/hypergan-signal-research/startup-baseline-v1`.

Each run performed 32 native updates and 5 monitor observations: 15 G observation
forwards total, 10 critic/10 pretrained observation forwards, 1280 pretrained
images including fixed-gray contexts. No directional or crossed-loss probes
ran. Source rollout elapsed 111.83s, including 62.45s inside native updates;
half-rate elapsed 96.64s, including 55.53s inside updates. Diagnostic time was
34.14s/29.79s; setup, audits and final restoration occupy the remainder. These
are single sequential wall-time measurements, not evidence that lower LR
makes training intrinsically faster. Neither reached the declared sustained
proxy-improvement onset. There was no algorithm-fitting probe cost for these
explicit controls; future algorithms must include their probe cost.

Both restored their complete disposable trainer state. Original config bytes
were unchanged, and protected pretrained/frozen parameter and buffer hashes
matched before, after and after restoration. No training checkpoint was saved.
These controls do not warrant extending to 512 updates merely to repeat their
failure. A promising future solution needs a separately declared longer test
because earlier failures appeared around 350–500 updates.

The older 1e-4 report used extra response/crossed probes and lacked explicit
bank identities. It stays historical rather than joining this matched table.
This protocol does not imply bitwise deterministic CUDA trajectories; the
configured backend permits nondeterministic kernels. Matching recorded
conditions also does not hash every training-data pixel: data artifacts remain
external dependencies identified by the source configuration and manifest.

## Next experiment

Use the existing layer-response evidence to form a bounded proposal. Check
the four influential early FFN down-projection weights jointly with the rest
of G, including interaction terms and unused finite probe points. Large
response alone does not justify suppression: those weights also contribute
substantial local descent. Save the formula, evidence and tuning solution
before evaluating it through this protocol. Keep negative curvature and
unresolved evidence as explicit abstentions rather than inventing a rate floor.
