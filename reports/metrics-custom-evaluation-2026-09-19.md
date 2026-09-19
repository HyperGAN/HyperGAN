# Custom metric factories and manual snapshot evaluation, 2026-09-19

This M4 slice implements ordinary Python scalar factories and explicit manual
snapshot evaluation over the [metrics research](metrics-first-research-2026-09-19.md)
backbone. Automatic snapshot schedules, asynchronous evaluation and a built-in
FID adapter remain unsupported; configuring an unsupported schedule fails
instead of being ignored. The new [configuration guide](../docs/configuration.md#custom-metrics-and-explicit-snapshot-evaluation)
records the supported contracts and examples.

## Implemented contracts

`metrics.custom.<id>` defines an importable factory, constructor arguments,
explicit input bindings, mode, cadence/deadline and failure policy. Structural
validation executes no factory and imports no numerical runtime. Bounded runtime
preflight constructs the factory and validates `describe()` before creating a
training attempt. Catalogs pin source-module hashes, declared output meaning and
specification. Changing a factory or its description changes observation
identity, while observation-only changes can still resume numerical state.

Scalar `evaluate(*, context, **inputs)` receives only selected detached primitive
update values and immutable context in a fresh group-free worker. The independent
broker owns deadlines and cleanup; a plugin never receives the live trainer,
modules, data sampler or RNG. Mandatory numerical checks still run independently.
One synchronous delivery is outstanding at a time; instances do not retain
state across calls. Optional failures publish a reason and disable the metric
for that attempt. Required failures fail the attempt; because the numerical
update already completed, the observed step may exceed the durable checkpoint
without a complete train event. Recovery replays from the durable checkpoint.
This boundary has an exact state-comparison regression.

The snapshot service takes a terminal run and an explicitly selected configured
metric. It holds the run lock and pins an immutable SHA256-checked EMA inference
bundle from this run's attempts directory, so this run cannot train or resume
concurrently. The snapshot copy is bounded while copying, parent/file symlinks
are rejected, and the checksum is a bounded ordinary file. No training object
crosses the worker boundary. The worker constructs its own graph and prior,
separate explicitly configured evaluation dataset, independent data/prior/global
RNG and bounded sample iterator. Custom data supplies `resume_identity()`; the
protocol records actual dataset/split/preprocessing identity and source hashes.
The metric must consume its complete declared sample count.

Snapshot `evaluate(*, batches, context)` returns a finite scalar or bounded fixed
histogram. The built-in examples provide pixel-weighted RGB mean/population-spread
distance and fixed-bin pooled RGB histogram probability differences. They
validate declared RGB ranges and shape; they do not claim semantic or spatial
image quality. The generic protocol accepts other modalities without depending
on these RGB examples.

Each evaluation publishes its own immutable event under
`metrics/evaluations/<id>/events.jsonl`, plus a result receipt and an atomic
`stream.json` registration. It never appends to training events. Successful
results identify the evaluated source attempt/step, snapshot digest, EMA choice,
factory/runtime sources, data identity, sample count, seed and protocol digest.
Failed results expose a reason and explicitly mark unknown source position;
they never create a fake step-zero metric value. Scalar values use `metrics` and
histograms use `distributions`. The standalone viewer discovers these streams on load, on reconnect and via
`stream_added` after training. A separate evaluation shelf loads each source's
own catalog and displays scalar/histogram results, source attempt/step, failed
status, protocol provenance, plots, exact-value accessible tables and raw exports.
Results never enter training curves or inherit the current training catalog.
Plots and tables allocate on expansion; result loads are serialized and bounded
by the 64-stream inventory. Live evaluation display does not run a server reducer.
General multi-evaluation aggregate projections remain a future extension.

Completed events and receipts precede stream registration. A later explicit
evaluation recovers an interrupted registration without recomputing the result.
An abandoned in-flight evaluation becomes an explicit failed stream and releases
its pinned copy; restarting uses a new evaluation ID and complete protocol seed,
not a silently resumed partial accumulator. There is no queue and no automatic
retry. Broker/host failure and descendants deliberately created by trusted plugin
code are outside the managed-process guarantee; arbitrary plugin allocations
are not a security sandbox.

The CLI integration helper exposes `evaluate RUN --metric ID [--config FILE]
[--bundle PATH]` and the same torch-free host API. The parser/dispatch is integrated into the shared CLI. GPU is the default evaluation device; CPU
use is explicit. This slice uses no implicit dataset/weight downloads and adds
no third-party dependency.

## Validation and cost

Source validation used isolated environments
`/tmp/hypergan-metrics-custom-verify` and
`/tmp/hypergan-metrics-custom-cuda-verify`, reusing existing pinned dependency
installations without modifying them. Final installed-package, CI and PR/merge
checks belong to coordinator integration.

- The complete new CPU configuration, file recovery, scalar and snapshot suite
  passed **26 tests in 94.59 seconds**. Tests cover structural rejection,
  per-metric cadence, factory source changes on resume, required/optional failure,
  native-blocked timeout and worker reaping, exact RNG/numerical isolation,
  pinned earlier-snapshot evaluation, repeated protocol parity, full iterator
  consumption, bounded copy, abandoned receipts and interrupted registration.
- Additional internal two-process accumulated CPU acceptance passed **1 test in
  7.80 seconds**, comparing every saved rank-state item with custom metrics on/off
  while proving the coordinator remains Torch-free. Missing factories and
  incompatible descriptors failed before run mutation: **2 tests in 6.80 seconds**.
- Actual native CUDA custom-scalar RNG isolation and CUDA manual snapshot
  repeatability passed **2 tests in 56.61 seconds**. After file-recovery
  hardening, complete two-GPU accumulated custom-metric state parity plus the
  manual CUDA snapshot gate passed **2 tests in 42.55 seconds**. CPU/default
  fixture checks remained separate; no hardware test was skipped.
- Five lightweight fresh-worker scalar calls measured 0.183–0.199 seconds,
  median **0.194 seconds**, on this workstation. This is explicit opt-in cost,
  not a default-metrics overhead result or throughput guarantee. A main module
  that imports Torch during process bootstrap costs more. Choose a cadence that
  amortizes worker startup; built-in scalar metrics launch no worker.
- Focused logs: `/tmp/hypergan-custom-final-cpu.log`,
  `/tmp/hypergan-custom-replicated.log`, `/tmp/hypergan-custom-cuda.log`,
  `/tmp/hypergan-custom-final-cuda.log` and `/tmp/hypergan-custom-cost.json`.
  The initial timeout fixture used a total deadline shorter than its Torch-heavy
  main-module bootstrap and correctly failed preflight; it was corrected to
  test an actual evaluator timeout. A test's in-process no-Torch assertion was
  also moved to an isolated subprocess because reference test collection imports
  Torch before foundation tests. These were fixture corrections, not skips.

Integrated API/browser acceptance used a separate source environment
`/tmp/hypergan-evaluation-ui-py312` without modifying installed-package validation
environments. **24 API/browser tests passed in 25.07 seconds**, including scalar
and histogram results whose catalogs differ from the training catalog, late
registration while training curves are unselected, failed source position,
accessible plots/tables, raw API exports, and discovery on reload. A separate
actual CPU training/evaluation/Chromium run produced two earlier-snapshot scalar
results at step 1, a 32-bin current-snapshot histogram at step 2, and an explicit
partial-iterator failure; all four appeared with no browser errors. Its receipt
is preserved in [the committed acceptance receipt](metrics-evaluation-browser-2026-09-19.json),
including source/event/catalog/protocol identities. The external screenshot is
`/tmp/hypergan-evaluation-browser-proof.png`. This proof explicitly prepares the
training projection before starting the read-only HTTP server. Source tests use exact M4 protocol
fixtures; this additional local proof exercises the actual supervised evaluator.

After integrating the standalone server's bootstrap/replay fixes, the combined
API/browser suite passed **29 tests in 28.11 seconds**. A later targeted run found
an artifact notification race during SSE replacement: a sample published after
the old connection closes but before the new subscription could be missed.
Refreshing the artifact inventory on `ready` fixes it without polling. A
regression now publishes exactly in that gap; the final browser suites passed
**10 tests in 21.73 seconds**, and the bundled JavaScript reproducibility check
passed. The actual CPU/evaluator/browser proof was rerun successfully against
that final source. Independent QC found no blocking protocol, isolation,
resource-bound or UI issues.

Public source cursors now carry logical run/stream/generation identity instead
of local path/inode identity. Copied-run replay, changed generation, consumed
boundary mutation, and inventory overflow preserving live manifest updates have
API regressions. Those backend fixes belong to the standalone-server milestone
and were separated as commit `40c78adf` for integration. The 64-stream cap remains
explicit; overflow reports a discovery error while existing streams continue.

No FID backend, pretrained weights, asynchronous resource allocation, automatic
snapshot scheduler, partial-job resume, real multi-host qualification, paid
compute or release is claimed. A future FID adapter must pin its implementation,
explicit local weight bytes, preprocessing, sample protocol and cache identity.
