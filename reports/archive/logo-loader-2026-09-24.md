# Logo loader parallelism and prefetch

The 128px logo recipe was spending most of each update in serial loading.
An isolated CPU benchmark loaded the same sequence of 64-image logo batches,
using the pinned recipe manifest and its unchanged preprocessing. No training
experiment, new seed comparison, or live-run modification was performed.

The implementation uses four persistent CPU threads by default and one batch
of lookahead within the existing permutation. It assembles and normalizes the
CPU tensor once per batch. Workers do not import Torch or use RNG. Source bytes
and containment are verified again when prefetched pixels are consumed.

The training split contains 404,757 entries: a complete 128px RGB uint8 cache
would require 18.53 GiB before overhead. Instead, the default lookahead holds
only 3 MiB of pixels per batch-64 loader, plus decoder working memory.

## Isolated timing

Environment: existing `training-runs/transgan-128-env`, CPU only,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. Both GPU training jobs continued
running. Each case requested 14 batches with the same caller-owned RNG state;
the median excludes the first two calls. Cases ran sequentially, so filesystem
caching and machine load can affect comparisons. These are loader measurements,
not measured end-to-end training speedups.

| Decoder threads | Prefetched batches | Simulated compute between requests | Median time inside loader |
| --- | --- | --- | --- |
| 0 | 0 | 0 ms | 289.0 ms |
| 4 | 0 | 0 ms | 127.9 ms |
| 8 | 0 | 0 ms | 121.5 ms |
| 4 | 1 | 50 ms | 86.8 ms |
| 8 | 1 | 50 ms | 87.7 ms |

A repeat after the final decoder warning-filter changes measured 267.4 ms
serial, 129.0/121.9 ms with four/eight threads, and 98.3/95.0 ms with
four/eight threads plus lookahead and the same 50 ms simulated compute.
That repeat overlapped CPU recovery tests as well as the two training jobs.
Across these checks, four threads cut batch loading to about 128–129 ms;
prefetch reduced the exposed wait to about 87–98 ms under simulated overlap.

Four threads were selected because eight gave little additional benefit. A
50 ms sleep simulates an opportunity to overlap CPU work with GPU computation;
it does not measure actual GPU overlap or account for training-thread contention.
Actual steps/s should be checked after the user restarts the logo job.

## Recovery and validation

Thread queues are disposable derived state and excluded from sampler state and
trainer deep copies. Lookahead never generates a new epoch, advances a data RNG,
or changes the sampler cursor. Failures drain outstanding work and restore both
sampler and RNG. Content hashes and symlink checks remain enforced, including
files changed after prefetch. Pillow decompression limits are checked explicitly
without changing process-global warning filters in decoder threads.

Tests cover serial/threaded bitwise pixel, label, sampler and RNG parity, changing
batch sizes and epoch crossings, concurrent decoding and background lookahead,
restore with pending work, copied checkpoint candidates, changed prefetched
files/symlinks, and full training checkpoint continuation.

76 focused tests passed: image data, colorization data/metrics, worker behavior,
image training recovery, checkpoint compatibility, data CLI, and the two-rank
image checkpoint continuation test. The distributed test validates trainer
deep-copy compatibility and exact resumed rank state. Ruff's fatal-error checks
passed for changed Python files; the new worker test file passes the full lint
selection. Existing unrelated style findings in older files were left alone.

Defaults apply to unchanged existing configs on process restart. Configured G/D,
optimizer, independent phase draws, preprocessing, sampling and seeds are unchanged.
