# 64px screen: both rate pairs fail startup

The source rates and the halved rates both fail the 32-update screen. This is
a separate resolution control from the 128px leaderboard. No tuning solution
is promoted. The live EMA-preview inspection remains a different measurement.

| Solution | G / D LR | Saturation at 32 | Diversity retained | DINO MMD² initial → final | Sustained proxy improvement |
| --- | --- | --- | --- | --- | --- |
| Source configuration | .0002 / .0002 | 100% | ~0 | .28730 → .48519 | Not observed |
| Half rates | .0001 / .0001 | 71.26% | 39.3% | .28730 → .40064 | Not observed |

See [leaderboard.md](leaderboard.md) and [leaderboard.json](leaderboard.json).
The real monitor bank is 22.57% saturated, with between-sample RMS .71824.
Initial generated diversity is .58408 on both rows. Source diversity at step
32 is about 1e-7. Half-rate diversity ends at .22975.

## Trajectory

Online generator, replayed particle IDs, one B64 monitor bank. The feature
protocol is `online_dinov3_block11_spatial_mean_candidate_only_poly3_64px_4x4`
(64px image, native 4×4 patch grid, final 384-channel spatial mean).

| Step | Source saturation | Source diversity | Source MMD² | Half saturation | Half diversity | Half MMD² |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.62% | .58408 | .28730 | 2.62% | .58408 | .28730 |
| 1 | 56.39% | .28649 | .32320 | 31.51% | .40735 | .32568 |
| 8 | 88.56% | .12550 | .50424 | 90.02% | .11059 | .43507 |
| 16 | 100% | ~0 | .48519 | 80.90% | .16070 | .40042 |
| 32 | 100% | ~0 | .48519 | 71.26% | .22975 | .40064 |

Source pre-tanh RMS rises from 1.183 to 23.56 and the mean tanh derivative
falls from .547 to about 1e-8. The outputs are saturated, not a preserved
gray initialization. Fixed-latent samples end in the same place: source
saturation 100% and half-rate 71.26%. At the final source generator, swapping
the initial prior for the current prior changes outputs by about 1e-8 RMS.
Half-rate prior swap changes outputs by .0111 RMS. Neither row shows a
sustained drop of the feature proxy below its step-0 value.

Half rates keep more diversity and a smaller feature-distance increase than
the source rates. They still saturate well above the real bank and lose most
of the initial sample variation. No winner is assigned. Neither control is
extended to 512 updates.

## Reproduction and audit

Measured commit `c7c085bd8de2cb72fe6da2cee00e12749807fb8e`. The recorded
source is dirty, so leaderboard identity stays on that exact source record
rather than an evaluator-tree digest. The untracked file in this worktree
during the run was the grouped-update contract note. The benchmark does not
import it. Both rows share that source record, seed 25002, config SHA256
`9b70e9f9a449f5744d6517124557dfb26a230b61ede8621a94a062d50e115f61`, initial
parameter hash `e0476f25601134f8…`, and monitor-bank hash `399828f2a4ffe535…`.
GPU 1 was selected by UUID. The effective fingerprint differs from the
original because the rollout sets the device to `cuda:0` on that visible GPU.

Manifest: [transgan-64-screen.json](../../configs/transgan-64-screen.json).
Solutions: [source.json](../../configs/solutions/source.json) and
[half-rates.json](../../configs/solutions/half-rates.json). Raw artifacts
also remain at `/mnt/ml7tb/hypergan-signal-research/startup-baseline-64-v1`.

Each run completed 32 native updates and 5 monitor observations. Source
elapsed 76.48s, of which 34.93s was inside native updates. Half-rate elapsed
68.18s, of which 32.90s was inside updates. No directional probe ran. There
was no algorithm-fitting cost. Both restored trainer state. Protected
pretrained weights and buffers matched before, after, and after restoration.
Original config bytes were unchanged. No training checkpoint was saved.

CUDA kernels are nondeterministic under the configured backend. These rows
are not a matched comparison with the 128px screen: the image size, patch
grid, feature protocol, parameter shapes, and initial-parameter hash differ.
