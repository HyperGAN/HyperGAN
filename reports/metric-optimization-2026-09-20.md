# Metric optimization audit, 2026-09-20

Training takes priority over observation. An observer running in another process
is insufficient isolation if the training thread waits for that process. This
audit traces the complete update → scalar transfer → publication → custom worker
→ projection/viewer path, plus periodic previews. It preserves complete-update
validation and checkpoint recovery instead of treating those correctness checks
as optional metrics.

The fixes remove ordinary per-update waits for plugins, event/status disk writes,
live control reads, replicated callbacks and preview rendering. Observation now
uses bounded background work and reports overload explicitly. On the actual
CIFAR fixture, measured controller/observation overhead was **1.16%** (95%
interval **0.41–1.92%**), compared with a baseline point estimate of 2.00%. The
custom-worker condition added **0.66%** with 75% of scheduled samples explicitly
skipped while busy. Complete saved training state matched in every trial and
across versions. These results do **not** establish zero overhead or a 1% bound.

## Baseline and method

Initial local, fetched and GitHub `develop` all resolved to
`74aab6c6ba591af30e5d1583cccad5956e75d364`; feedback integration
[PR #343](https://github.com/HyperGAN/HyperGAN/pull/343) was merged and no PRs
were open. The resurrection ledger and linked plan were read first. This is a
new performance audit, not a repeat of the historical repository audit.

Three subagents independently traced and implemented the scalar-transfer,
custom-worker and observation-I/O slices in external worktrees. The coordinator
reviews their changes, measures GPU behavior, and integrates passing small PRs
into `develop`. This report records both structural guarantees and experimental
limits; no zero-overhead claim follows from a nonsignificant timing result.

Only physical GPU 0 (RTX A6000, UUID
`GPU-ed080e41-3193-3755-6756-f3d46c433331`) is available for these experiments.
Every CUDA validation process is restricted to that UUID with
`CUDA_VISIBLE_DEVICES`; its logical device is `cuda:0`. GPU 1 is running the
owner's training and is excluded. Existing jobs on either card and the owner's
frozen package environment are untouched. Dedicated baseline and candidate validation wheels are
installed into separate `/tmp/hypergan-metric-opt-*` environments; existing dependencies are read
through explicit site-package paths, without changing those environments.

## Findings at the audited baseline

| Path | Finding | Required change or boundary |
| --- | --- | --- |
| Native update scalars | `ReferenceTrainer.update` converts each detached CUDA scalar separately to a Python number. | Stage detached values into reusable pinned host buffers, then synchronize once; preserve precision and finite validation. Device packing was measured and rejected. |
| Replicated update scalars | Each rank converts metrics to host numbers, creates another GPU tensor for reduction, then transfers results again. | Consolidate scalar-read synchronization; retain the existing float64 reduction and numerical agreement. |
| Custom scalar metrics | `ScalarMetrics.evaluate` starts and waits for a fresh supervised process for each due metric, including startup and shutdown. | Nonblocking bounded submission and result polling; retain source-step identity, explicit busy/drop states and configured failure policy. |
| Event journal | Every update opens, serializes, writes and flushes its event synchronously. | Bounded background publication, explicit overload evidence, ordered accepted events and checkpoint drain barriers. |
| Status manifests | Every update durably rewrites both run and attempt manifests, including file and directory fsync. | Coalesce ordinary progress in a worker; preserve durable lifecycle, reservation and checkpoint barriers. |
| Checkpoint requests | Directory/path/queue inspection occurs every update. | Bound poll frequency while honoring requests at complete update boundaries and finalization. |
| Native previews | CPU EMA copy, generator forward, grid encoding and publication run synchronously in the training process. | Capture an immutable snapshot at a complete boundary; render/publish off the training path with bounded outstanding work. |
| Replicated previews | Rendering has its own CPU worker, but the controller waits for it. | Background renderer dispatch; training resumes after the necessary snapshot capture. |
| CLI progress | Native adapter snapshots/restores Python, NumPy and visible CUDA RNG around even the trusted bounded CLI output sink. | Bypass numerical-state isolation only for the exact internal sink type. |
| Arbitrary callbacks | Native Python `on_event` is an inline control hook; replicated callbacks are isolated but synchronously awaited. | Distinguish these APIs from passive metrics; do not move closures into shared-RNG threads or claim they are nonblocking. |
| Viewer/projector | Separate processes already consume immutable file streams; subscriber queues and history work are bounded. | Preserve isolation and measure contention; separate processes still share CPU, memory bandwidth and storage. |
| Snapshot/FID evaluation | Manual evaluation uses an independent immutable snapshot and explicit evaluation device. | No automatic training-GPU evaluation; explicitly launched work on the same GPU can contend with training. |

Correctness barriers remain deliberate: complete-update finite checks,
replicated collectives, immutable snapshot capture, and checkpoint save/fsync
cannot honestly be described as cost-free observation. Optional rendering and
plugins must not hold up the next update. Queues must be bounded and overload
visible; unbounded buffering merely postpones a training failure.

## Validation method

The existing GPU proof compares metrics disabled, default metrics, server with
zero consumers, and server with five actual HTTP SSE consumers. It compares
complete saved numerical/RNG/data state and uses four rotated-order blocks
with 64 warmup plus 1,024 measured updates per trial. The prior
[performance report](metrics-observation-performance-2026-09-19.md) did not
establish its proposed 1%/2% targets; those targets remain unproven until this
report supplies new evidence.

The first baseline invocation exposed a stale harness assumption: the viewer now
defaults to no authentication, while the benchmark expects token credentials.
The harness now explicitly requests token authentication. The failed startup
and corrected invocation are retained; no training result is inferred from it.

Durable commands, logs, wheel/source hashes, benchmark receipts and PR merge
receipts are stored under
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-20-metric-optimization/`.
Final acceptance and protected merge receipts are recorded below.
No paid compute, release, GPU-1 experiment or old-checkpoint migration is included.

## Reviewed implementation slices

- [#344](https://github.com/HyperGAN/HyperGAN/pull/344): scalar transport. The first
  candidate reduced nine device-to-host copies to one, but an actual CUDA probe
  rejected it: median transport increased from 83.76 µs to 233.06 µs. The receipt
  `scalar_cuda_probe.json` preserves this regression. The accepted implementation instead stages nine asynchronous scalar copies into
  reusable pinned host buffers behind one stream fence, without GPU packing
  kernels. Median helper latency fell from 80.75 to 63.27 µs; p99 fell from
  112.33 to 79.29 µs. This is a transport microbenchmark, not end-to-end speedup.
- [#345](https://github.com/HyperGAN/HyperGAN/pull/345): bounded background event
  persistence and coalesced progress manifests. At most 256 pending events and
  8 MiB of conservative encoded size are admitted, plus one active bounded row.
  Overload records counts/ranges by event kind. Accepted sequence numbers remain
  contiguous, and checkpoint barriers record a current-step marker if the latest
  completed observation belongs to an older step. Unicode surrogate escaping and
  large integer sizes were included in the byte-bound review.
- [#346](https://github.com/HyperGAN/HyperGAN/pull/346): custom scalar submission,
  asynchronous source-step results, explicit busy drops, CPU-only lower-priority
  workers, finite admission deadlines and cancellation/reaping. Required failures
  remain failures, including required work cancelled by a stop request. Optional
  cancellation is recorded. Stop does not wait out a configured one-hour plugin
  deadline. Normal completion drains admitted observations before the final
  checkpoint frontier.
- [#348](https://github.com/HyperGAN/HyperGAN/pull/348): bounded asynchronous
  replicated callbacks with accepted/completed/dropped/failed counters. Terminal
  delivery persists counters and failures durably; the review found and fixed
  a throttle that otherwise omitted this terminal metadata. Native `on_event`
  remains the explicitly synchronous Python control-hook API, including closures.
- [#349](https://github.com/HyperGAN/HyperGAN/pull/349): background console-policy
  and checkpoint-request readers. Ordinary updates consume cached results.
  Startup settings/overrides and fresh checkpoint/final control scans remain
  explicit boundaries. Blocked optional reader shutdown does not hold training.
- [#350](https://github.com/HyperGAN/HyperGAN/pull/350): one outstanding immutable
  periodic preview, rendered and published by a CPU process. Busy previews are
  skipped before reserving another sequence or copying another model. Failures
  remain visible. Immutable GPU-state capture and durable sequence reservation remain boundary
  costs. Snapshot serialization is addressed by the next slice.
- [#351](https://github.com/HyperGAN/HyperGAN/pull/351): native snapshot directory
  creation, serialization, fsync and hashing move to the supervisor. The update
  boundary freezes owned CPU tensors and plain containers, so the storage worker
  cannot read live training state or execute custom serialization hooks. The
  single outstanding snapshot has explicit memory/file limits, cancellation and
  a deadline covering storage and rendering. Replicated rank-zero file handoff
  remains a synchronous transport boundary.
- [#347](https://github.com/HyperGAN/HyperGAN/pull/347): coordinator integration,
  report and reproducible benchmark fixes. The exact internal CLI sink bypasses
  native numerical RNG snapshots and replicated per-event health round trips;
  arbitrary Python callbacks do not receive that privilege.

The custom dispatcher CPU probe recorded five original synchronous invocations
at **198.5–261.2 ms** each. With one observation outstanding, 1,000 new submit/poll
calls measured **1.54 µs median, 2.87 µs p99**, with one accepted observation and
999 explicitly dropped samples. The final result retained source step 1. This
measures submission overhead under overload, not plugin throughput or GPU
training speed; `workers/dispatcher-benchmark.json` preserves the receipt.

Fault-injection coverage includes blocked event writes while ten updates finish,
queue saturation and explicit gaps, durable-prefix validation, late custom
results at their original step, same-step final checkpoint refresh, failed
publication, acknowledgement retry without duplicate saves, blocked control-file
reads, real plugin timeout/cancellation and process reaping. Installed-package and CUDA acceptance receipts are summarized below.

## Performance receipts and interpretation

The bare condition calls the numerical execution adapter directly. It still
performs mandatory scalar extraction and finite validation. Therefore
`metrics/bare` measures the controller and observation together, while
`metrics/none` compares optional metric collection inside the same controller.
The measured window excludes startup, final checkpoint/export and terminal
worker drains. These timing fixtures have no periodic checkpoints or previews;
preview correctness and blocked-storage progress are covered separately.

The synthetic fixture uses 1,088 updates, discarding 64 warmup updates, with
four blocks of five conditions (20 trials). The CIFAR-10 fixture uses the owner's
actual image recipe, batch size 64 and 16,384-component MoG with pretrained
ResNet features, using cached data/weights. It measures 96 updates after 32 warmup
updates in each 128-update trial. The baseline has four bare/metrics pairs;
the candidate also schedules the example custom ratio metric every update.
All conditions compare complete saved numerical state, including optimizer,
prior, EMA, RNG, data cursor, last batch and registered buffers.

| Baseline comparison | Paired elapsed overhead | 95% interval |
| --- | ---: | ---: |
| Synthetic metrics / bare | +9.03% | −6.23% to +26.78% |
| Synthetic metrics / metrics disabled | +0.71% | −21.40% to +29.04% |
| Synthetic five viewers / zero viewers | +0.42% | −6.75% to +8.14% |
| CIFAR metrics / bare | +2.00% | +0.02% to +4.03% |

The integrated candidate at `2c749af9` completed all 20 synthetic trials. Every
saved numerical state matched both the other conditions and the baseline.

| Candidate comparison | Paired elapsed overhead | 95% interval |
| --- | ---: | ---: |
| Synthetic metrics / bare | +7.05% | +0.01% to +14.58% |
| Synthetic metrics / metrics disabled | +1.63% | −4.84% to +8.54% |
| Synthetic five viewers / zero viewers | +0.76% | −6.89% to +9.05% |
| Synthetic server plus five viewers / metrics | +0.62% | −2.17% to +3.50% |
| CIFAR metrics / bare | +1.16% | +0.41% to +1.92% |
| CIFAR custom worker / metrics | +0.66% | −0.21% to +1.53% |

All 12 candidate CIFAR trials also completed with identical numerical state,
matching the baseline. In each custom trial, 128 scheduled observations produced
32 admitted/completed results and 96 explicit busy drops. Across four trials,
all 128 results match their admitted source step and that step's loss ratio;
there were no event-journal gaps or observation errors. This is **75% sampling
loss by design**, not a claim of 128 plugin evaluations per trial. Manifests,
events, recipes, source/result checks and hashes are preserved in
`final-image-custom-runs/` and `final-image-custom-observations.json`.

These measurements still do **not** establish a 1% metrics or 2% viewer upper
bound. The small synthetic model exposes controller overhead; moving waits out
of the update path does not make bookkeeping free.

Intervals use paired log elapsed ratios and Student-t with four blocks. This
small sample relies on approximate independence/normality and does not establish
equivalence across workloads. Five conditions in four rotated blocks are not a
fully counterbalanced design. GPU 0 has an unrelated low-activity allocation;
GPU 1's owner job and other CPU activity share host resources. Baseline and
candidate runs occur at different wall-clock times, so their absolute throughput
difference is not a controlled causal speedup estimate. Negative overhead is
measurement variation, not free acceleration from observation. Numerical state
equality establishes noninterference for these fixtures, not zero performance
cost or GAN quality. The baseline did not establish a 1% metrics or 2% viewer
upper bound.

## Remaining explicit boundaries

- Training still waits for numerical validation, replicated reductions and
  checkpoint durability. Ordinary observation does not bypass these checks.
- Snapshot capture copies current state at a completed update boundary. Native
  snapshot storage/rendering is asynchronous; replicated rank-zero snapshot file
  transport still serializes, fsyncs and hashes before the ranks resume.
- Native Python `on_event` is a synchronous control hook, including closures and
  RNG isolation. Use the metrics worker API for passive custom observation.
- Optional workers share host CPU, memory and storage bandwidth. Bounded queues,
  one-thread numerical-library defaults and reduced POSIX priority minimize but cannot
  eliminate contention. Busy drops and observation gaps remain visible.
- Explicit manual snapshot/FID evaluation can use a requested GPU and compete
  with training if the caller chooses the same card. This audit does not silently
  schedule those evaluations on the training GPU.
- OS calls stuck in the kernel cannot be forcibly cancelled inside a thread.
  Optional readers can be abandoned; preview storage has bounded cleanup and
  reports failure. Durable checkpoint/event barriers may wait for storage.
- GPU 1 is reserved for the owner, so this change has native CUDA and replicated
  CPU validation, not a new two-GPU or multi-host qualification. Existing
  distributed qualification must not be treated as proof for these new paths.

## Acceptance and integration

Scalar slice [#344](https://github.com/HyperGAN/HyperGAN/pull/344) passed all
protected checks at `d9075cee` and merged normally into `develop` as
`895f9c7681f81569627f07d243712f75ed408b20` on 2026-09-20. No branch-protection
bypass or release publication was used. Other slice heads are preserved in the
integration history; acceptance is judged against the combined implementation,
including later fixes to stale synchronous-observation assertions.

The first broad installed run at `7c809929` passed 808 tests and failed four. Two
preview failure-boundary cases exposed exception-severity handling that the
review fixed; preview-count and manual-save assertions still assumed immediate
synchronous completion. Their replacements require explicit busy skips, original
source-step identity, safe complete-update checkpoint boundaries, one coalesced
save and acknowledgement retry. None was skipped or weakened to ignore a failed
operation. The original failure log remains in the evidence directory.

Actual SIGTERM regressions use native-blocked custom metric and callback workers
with one-hour configured deadlines. They verify bounded cancellation/reaping,
source-step evidence and preservation of an existing numerical failure. Preview
regressions cover immutable snapshots, blocked disk while updates continue,
storage/render deadlines, signal cancellation, late results and cleanup failures.
The callback follow-up passed 51 focused checks, including real two-process CPU
execution.

Installed CPU acceptance at `2c749af9` passed **514 tests** in 152.77 seconds; all
56 installed runtime Python files match the archived source. Eleven selected
native CUDA tests passed, including save/resume, preview-enabled recovery,
scalar-worker isolation and one-fence scalar transport. The additional manual
CUDA snapshot fixture first exposed a missing explicit cuBLAS determinism setting,
then a real unsupported deterministic CUDA `histc` call.

Follow-up [#352](https://github.com/HyperGAN/HyperGAN/pull/352) publishes complete
viewer credentials atomically without replacing existing paths, fixing a Windows
readiness race. Its installed web/observer suite passed 55 tests. Callback timeout
fixtures now allow Windows process startup before asserting timeout and process
reaping. Follow-up [#353](https://github.com/HyperGAN/HyperGAN/pull/353) computes
example color-histogram aggregation on CPU inside the isolated evaluation worker,
preserving strict deterministic policy; four CPU snapshot tests passed. A test-only
cleanup also prevents missing-runtime fixtures from starting persistent viewers.
Only the verified test viewers were stopped.

The timing wheels correspond to `2c749af9`. Later changes remove one duplicate
terminal request scan, cancel replicated callbacks on terminal signals, publish
credentials atomically, fix manual histogram evaluation and clean up test viewers.
They do not change the measured native update/observation paths. Source archives,
wheel hashes and exact per-file identities distinguish these acceptance stages.

Reproduction commands (run outside the checkout with its dedicated installed
wheel and the evidence directory substituted for `EVIDENCE`):

```sh
CUDA_VISIBLE_DEVICES=GPU-ed080e41-3193-3755-6756-f3d46c433331 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/tmp/hypergan-metric-opt-candidate/bin/python \
/home/martyn/dev/hypergan/metric-optimization/scripts/metrics_training_proof.py run \
  --device cuda:0 --include-bare --steps 1088 --warmup 64 --repetitions 4 \
  --output EVIDENCE/final-with-bare.json

CUDA_VISIBLE_DEVICES=GPU-ed080e41-3193-3755-6756-f3d46c433331 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/tmp/hypergan-metric-opt-candidate/bin/python EVIDENCE/image_observation_probe.py \
  --custom --output EVIDENCE/final-image.json
```

The image probe requires the preserved local CIFAR recipe/data/weights; it records
the original/generated recipe hashes, runtime versions and package source hashes.
Existing output directories intentionally fail rather than overwrite receipts.
Use a fresh output path for a repeat experiment.

Final strict CUDA snapshot/repeatability and histogram policy checks passed
**2 tests** in 37.91 seconds against installed source `0955fa6f`, completing the
manual-evaluation failure follow-up. The first final-environment collection attempt
used system Python 3.14 with Python 3.12 dependency paths and collected no tests;
the corrected dedicated Python 3.12 environment preserves its separate passing
receipt. No dependency or owner environment was modified.

The integration PR preserves all ten reviewed PR heads (#344–#353), verified by
`git merge-base --is-ancestor`; only the exact combined head may pass the final
protected merge gate. `preserved-pr-heads.json`, `pr344-premerge.json` and
`pr344-merge.json` retain ancestry and the first completed protected merge. Final
GitHub check/merge receipts are stored alongside them; the linked integration PR
records the final merge commit and check history.

Final-source CPU test output, exact invocation, distribution hashes and installed
package identity are retained in `workers/final-0955fa6f/tests.log` and
`workers/final-0955fa6f/identity.json`. This covers all foundation tests plus
actual native/replicated signals, manual saves and recovery, callback/custom
workers, snapshot evaluation, CLI, previews and observation persistence.
