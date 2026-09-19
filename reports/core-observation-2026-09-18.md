# Core observation checkpoint — 2026-09-18

This slice finishes the local CPU observation foundation before a browser server or a distributed trainer is introduced. Three subagents divide event/request protocols, preview/trainer integration, and independent CPU collective groundwork; the coordinator integrates the CLI, tests installed distributions and reviews the changes for `develop`. No GPU execution, dataset download, cloud allocation or release publishing is part of this checkpoint.

## Delivered core workflow

- `events RUN_DIR` exposes bounded forward pages with opaque reconnect cursors. Complete versioned records preserve run/attempt identity, partial trailing writes do not advance the cursor, and stale boundary/file generations are rejected. It imports no numerical runtime.
- `checkpoint RUN_DIR` submits an attempt-bound, idempotent local request. `checkpoint --status ID` distinguishes pending submission from an acknowledged durable checkpoint. The trainer serializes saves only at complete D/G/EMA boundaries. Requests never silently carry into a resumed attempt.
- `train/resume --preview-every N --preview-keep N` publishes isolated EMA previews with bounded payloads and retained history. `--no-previews` disables periodic sampling. Monotonic artifact reservations survive older-checkpoint replay; final inference bundles and training checkpoints remain separate from preview retention.
- The CLI and Python API use the same files and protocol. The future viewer can consume them without constructing models, joining process groups or running numerical callbacks in an HTTP handler.

The [observation guide](../docs/observation.md) defines exact controls, bounds and limitations. Previews are numeric JSON, not image grids. Checkpoint submission is not execution: a request racing the last boundary can remain pending until a later attempt rejects its stale target. This trusted local filesystem interface does not supply remote authentication or whole-job supervision.

## Review and validation

Acceptance must exercise base-only installed commands, complete preview-on/off numerical state comparisons, interrupted/resumed stochastic components, immutable artifact retention, safe checkpoint acknowledgement and a live cross-process CLI request. Required CI tests the installed wheel on nine lightweight platform/Python combinations and the numerical CPU runtime on Linux. Exact commits, PRs and final test results are recorded in the execution ledger at integration.

Review separates numerical-state protection from arbitrary custom Python side effects. EMA graph/prior copies and copied conditioning inputs prevent ordinary in-place buffer/input changes from reaching training. CPU global RNG and named sampling streams are isolated. Custom constructors/forwards are trusted code; the payload bounds do not impose a hard compute/memory deadline on that code.

## Next satisfying cutpoint

- [ ] Merge this observation slice after independent review and installed-package/CI acceptance, then update the W1 ledger with evidence.
- [ ] Merge bounded CPU collective groundwork separately: compare global objectives, first/second derivatives and prior populations in two real Gloo processes. Label this numerical groundwork explicitly; it is not a training launcher.
- [ ] Implement one fixed-world-size CPU distributed update path with explicit G/prior/auxiliary ownership, alternating D/G reducers and accumulation semantics. Compare parameters, Adam state and EMA across complete updates against a controlled global reference.
- [ ] Add coordinated fixed-topology checkpoints, per-rank RNG/data state and whole-job failure/restart. Only complete rank sets establish a recoverable global update.
- [ ] Resolve the image-reference license/extraction gate and freeze its actual update, data and evaluation contracts.
- [ ] Test actual local two-GPU NCCL only after the CPU gates pass; prepare a separately agreed real two-node allocation afterward.

The standalone read-only browser server can follow W1 in parallel with distributed work. It remains optional and is not a prerequisite for proving numerical correctness. Image quality, cluster readiness, containers and release promotion remain open gates.
