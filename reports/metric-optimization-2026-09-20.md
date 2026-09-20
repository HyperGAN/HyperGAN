# Metric optimization audit, 2026-09-20

Training takes priority over observation. An observer running in another process
is insufficient isolation if the training thread waits for that process. This
audit traces the complete update → scalar transfer → publication → custom worker
→ projection/viewer path, plus periodic previews. It preserves complete-update
validation and checkpoint recovery instead of treating those correctness checks
as optional metrics.

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
frozen package environment are untouched. Dedicated validation wheels are
installed into `/tmp/hypergan-metric-opt-verify`; existing dependencies are read
through explicit site-package paths, without changing those environments.

## Findings at the audited baseline

| Path | Finding | Required change or boundary |
| --- | --- | --- |
| Native update scalars | `ReferenceTrainer.update` converts each detached CUDA scalar separately to a Python number. | Pack detached values on device and transfer once; preserve precision and finite validation. |
| Replicated update scalars | Each rank converts metrics to host numbers, creates another GPU tensor for reduction, then transfers results again. | Remove the unnecessary host round trip; keep global reduction and numerical agreement. |
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

## Validation in progress

The existing GPU proof compares metrics disabled, default metrics, server with
zero consumers, and server with five actual HTTP SSE consumers. It compares
complete saved numerical/RNG/data state and uses four counterbalanced blocks
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
Implementation and final acceptance results will be appended before handoff.
No paid compute, release, GPU-1 experiment or old-checkpoint migration is included.

## Reviewed implementation slices

- [#344](https://github.com/HyperGAN/HyperGAN/pull/344): scalar transport. The first
  candidate reduced nine device-to-host copies to one, but an actual CUDA probe
  rejected it: median transport increased from 83.76 µs to 233.06 µs. The receipt
  `scalar_cuda_probe.json` preserves this regression. Revised transport is being
  measured; fewer copies is not itself an acceptance criterion.
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
  remain visible. Native snapshot serialization is being moved to its supervisor
  as a follow-up; immutable GPU-state capture and durable sequence reservation
  remain costs that must be measured rather than hidden.
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
reads, real plugin timeout/cancellation and process reaping. Integrated installed
package and final CUDA receipts are pending at this report checkpoint.
