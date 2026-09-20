# Training workflow feedback implementation — 2026-09-20

This report closes the implementation work in [the feedback report](feedback-2026-09-20.md).
The reference CIFAR run and its frozen environment remain unchanged; the run was
stopped at step 41,000 when this work began. All training validation here uses
separate small fixtures. These changes do not extend the image-quality allocation
or qualify the actual CIFAR recipe for two GPUs.

## Reviewed slices

| Feedback | Implementation PR | Result |
| --- | --- | --- |
| 1. CLI cadence | [#342](https://github.com/HyperGAN/HyperGAN/pull/342) | Routine progress defaults to every 100 steps; `--progress-every N` and the live UI control persist per run. Metrics, previews and checkpoints retain independent cadences |
| 2. Tensor samples | [#336](https://github.com/HyperGAN/HyperGAN/pull/336) | Numeric previews handle the actual default 64×3×32×32 CIFAR sample, display 128 values, validate the tensor and retain a bounded download path. PNG display remains working |
| 3. Termination | [#341](https://github.com/HyperGAN/HyperGAN/pull/341), [#342](https://github.com/HyperGAN/HyperGAN/pull/342) | SIGINT/SIGTERM request a complete-update stop and checkpoint. Worker groups and console drains survive the first terminal-group signal. Repeated signals or the shutdown deadline force exit |
| 4. Bind/auth | [#337](https://github.com/HyperGAN/HyperGAN/pull/337) | Default bind is `0.0.0.0`; `--host`/`--server-host` configure it. Authentication defaults to none; `--auth token` explicitly enables the token flow |
| 5. Server lifetime | [#339](https://github.com/HyperGAN/HyperGAN/pull/339) | Automatic server/projector survive training completion, failure and termination. Resume reuses the incarnation; `server-status` discovers it and `stop-server` shuts it down |
| 6. Durable metric boundary | [#341](https://github.com/HyperGAN/HyperGAN/pull/341), [#340](https://github.com/HyperGAN/HyperGAN/pull/340), [#342](https://github.com/HyperGAN/HyperGAN/pull/342) | Checkpoints reference a synced event prefix. Restore validates its identity and hash. API/UI distinguish committed metrics from a lagging or unavailable projection |
| 7. FID presentation | [#338](https://github.com/HyperGAN/HyperGAN/pull/338) | Snapshot evaluations appear in the normal metric selector/charts, ordered by snapshot step with sparse markers. Protocols, definitions, attempts and repeated measurements remain distinct |

Three subagents implemented and cross-reviewed bounded slices in external
worktrees. The coordinator reviewed the code, resolved shared CLI/API/documentation
changes, and tested an installed combined wheel built through the source archive.
The integration retains each reviewed branch's commits and required CI; protected
merge receipts record the accepted head. No branch protection was bypassed.

## User workflow

For a newly created run under the updated installation:

```sh
hypergan train config.toml --run-dir runs/example --progress-every 100
hypergan server-status runs/example
hypergan resume runs/example
hypergan stop-server runs/example
```

The browser's **CLI progress every N steps** setting takes effect at a completed
update boundary and persists into resume. Settings are read at most four times
per second; a long update can delay application. Lifecycle, error and final-result
messages remain immediate best-effort output. The event journal is authoritative.
`--progress-json` applies the same cadence to routine terminal progress without
reducing the stored metrics.

`--server` requires successful viewer startup; optional startup remains asynchronous
and immediately prints discovery/stop commands. `--no-server` remains explicit
headless execution. Use `--server-host 127.0.0.1` to select loopback and `--auth token`
for authentication; standalone `serve` uses `--host`. Wildcard listen addresses are
not used as browser destinations. Active viewer bind/auth settings are inherited
on resume; conflicting explicit choices fail with a stop/restart instruction.

## Durability and interruption contract

The controller finishes the update and emits its configured measurements, syncs
the event prefix, writes the complete checkpoint payload, and then atomically
publishes the checkpoint reference. Metadata identifies the exact run, attempt,
step, event sequence, byte offset and SHA256. Checkpoint notifications follow the
commit and are not part of its measurement prefix. Incremental hashing avoids
rescanning all previous events at every checkpoint.

A crash before publication preserves the previous reference; a crash after
publication can leave the new valid checkpoint even if the run manifest is older.
Corrupt committed payload/events fail explicitly. Selecting an earlier intact
checkpoint is supported; uncommitted suffix events remain with attempt ancestry
rather than being silently treated as the resumed trajectory. No migration for
older runtime/checkpoint formats was added.

The consistency readout compares the built-in projection's source byte cursor
against the committed event boundary. It updates even after training stops.
It is server-observed projection progress, not an acknowledgement that a browser
has rendered every frame or a second validation of disk durability.

Graceful signal handling has a 30-second escalation timer and does not start
further optional preview/final inference work after observing the stop request. SIGKILL is not catchable.
Python handlers/watchdogs can be delayed by native extensions holding the GIL;
an external supervisor supplies a hard deadline. File/directory sync guarantees
remain subject to filesystem/platform behavior; Windows has no portable directory
fsync in this implementation. See [recovery](../docs/recovery.md).

## Acceptance evidence

- The installed CLI/browser walkthrough ran 105 updates with only the step-100
  training line and all 105 stored metric events. A UI change to five steps
  persisted into resume; the same viewer incarnation remained available. An
  explicit one-step override printed steps 106–110 after earlier-snapshot replay.
  Default no-auth browser entry, projection catch-up and explicit server stop passed.
- Chromium displayed the owner's real 4,131,357-byte `[64,3,32,32]` tensor,
  verified its first 128 numeric values and downloaded the exact original bytes.
  A read-only view of the actual run showed its four FIDs ordered 10k, 20k, 30k,
  40k in the chart/details/table, with training charts loaded and no browser errors.
- Native installed CUDA: **8 passed in 15.41 seconds**. Actual two-GPU public CLI
  and NCCL recovery: **7 passed in 178.20 seconds**, including exact complete
  state after resume/earlier selection, rank failure and coordinator death.
  These fixtures qualify the generic runtime changes, not distributed CIFAR training.
- Recovery-slice installed checks: **144 passed** in the broad pass and **125
  passed** after the final focused rebuild. Fault stages include event sync,
  payload writes, directory rename, reference publication and postcommit manifest
  failure, plus native/replicated signals and forced process death.
- Final archive/wheel hashes, integrated installed foundation/reference/web/
  reducer/browser results, focused settings/lifecycle corrections and exact-head
  protected CI results are retained with the acceptance and merge receipts.

Review and testing caught additional integration defects: browser preview bounds
excluded normal CIFAR tensors; the HTTP download filename overrode the HTML name;
Windows needed exclusive port reservation; an old CLI test assumed viewers die
with training; terminal-group signals could kill console output drains; and a
nonregular console settings file could block an update callback. Each received a
specific fix or corrected lifecycle assertion, without hiding failures or widening
numerical tolerances.

The original tensor symptom was not described precisely enough to prove its cause.
The reproduced numeric-preview limit is fixed and the owner's real final artifact
was exercised; this does not reinterpret the correctly displayed PNG as broken.

Durable artifacts, commands, wheel/source identities, test reports, browser
screenshots and merge receipts are retained under
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-20-feedback/`.
The recovery agent's additional publication/signal evidence is under the sibling
`2026-09-20-feedback-recovery/` directory. No paid compute or release occurred.
