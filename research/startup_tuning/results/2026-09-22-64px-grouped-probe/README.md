# 64px grouped first step: the four layers dominate, and the quadratic does not

The probe abstains. No layer multipliers, global rates, or initialization
scales were emitted, and no benchmark rollout was spent on that abstention.
There is no longer-run candidate.

The candidate group is the same four early FFN down-projections measured at
128px. At source rates 2e-4, on the actual first generator Adam step, both
fitting banks put exactly those four tensors in the top 4 by estimated squared
output response. Their share of the sum of per-tensor squared responses is
0.933 and 0.938. The predeclared bar was 0.70. The grouping is revalidated at
64px. The fifth tensor on both banks is a 16px attention projection, more than
an order of magnitude smaller.

That dominance is not a reason to shrink the group. Both groups have negative
curvature on both banks, so the contract stops at the curvature gate and does
not emit a stationary factor.

| Bank | aA | kA | aB | kB | kAB | Loss change of the full step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -0.10437 | -0.16716 | -0.09488 | -0.12288 | 0.14174 | -0.20253 |
| 1 | -0.10435 | -0.17263 | -0.08956 | -0.12666 | 0.13988 | -0.20368 |

Symmetric and exact group slopes are both negative: the full step descends,
and each group descends on its own axis. Negative curvature means the fitted
quadratic has no minimum in the positive-step direction. The cross term does
not repair that. No absolute value was taken, and no rate floor was applied.

Separately, the finite output change is not a sum of the two group changes.
The residual RMS is 0.757 and 0.750 of the joint-step RMS. Independent
multipliers would not be a decomposition of the output response. This is not
the gate that fired. The curvature gate already abstains.

The Adam proportionality record is also not the deciding gate. Group A’s
step matches `-lr g / (|g| + eps)` to a relative RMS of 1.04e-6, with 0.224%
of its slope mass on tiny gradients. Group B’s relative RMS is 3.54e-6, but
3.83% of its slope mass sits on elements with `|g| ≤ 1000 eps`, above the 1%
bar. A later passing curvature model would still have refused to treat B’s
factor as a learning-rate multiplier.

## Cost and identity

Clean commit `47389a555e603c3e26133a59dd597c926c32f532`, GPU 1 by UUID.
One native update, 14 generator phase-loss evaluations, 8 projection
backwards, 4 output forwards, 38.00 seconds. The monitor-bank hash equals the
64px baseline monitor bank
`399828f2a4ffe535…`. The two fitting banks are different hashes and were the
only banks used for ranking and the stencil. State was restored. Protected
pretrained weights and buffers stayed
`01d54beef4f137b8e502c67dc7896e6a619eef9fe2c1e67c0820aa906e49fa9c`.

A later check of the decision code corrected gate order and failure labels.
Reapplying that decider to this same probe still abstains at curvature, because
both fitting banks fail that gate. The recorded losses were not refit.

The raw report is [probe.json](probe.json). The same file remains at
`/mnt/ml7tb/hypergan-signal-research/grouped-first-g-64-v1.json`. This is not
a leaderboard row: the leaderboard is the two failed 64px controls. The
source configuration remains the control, and it is already saturated by step
16.
