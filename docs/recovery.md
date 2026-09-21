# Recover training runs

Native CPU/CUDA and local replicated execution have separate **training checkpoints** and **EMA inference bundles**. A training checkpoint restores the numerical run. An inference bundle loads the generator and prior for sampling; it cannot resume training. Actual image-recipe multi-GPU and real multi-host recovery remain separate qualification gates.

Create a project and stop after two updates without changing its five-update learning-rate schedule:

```sh
hypergan new demo
hypergan train demo --run-dir runs/demo --checkpoint-every 1 --stop-after-steps 2
hypergan inspect runs/demo
hypergan train demo --run-dir runs/demo
hypergan sample runs/demo --count 16
```

`train CONFIG --run-dir RUN_DIR` creates a new run when the directory is absent and resumes its latest complete checkpoint when it already exists. The supplied resolved configuration must match the saved run; comments and formatting do not affect that comparison. An unrelated directory, incomplete run without a full checkpoint, or invalid checkpoint fails without starting a fresh experiment in that directory. Use a new run directory for changed training settings. Repeated `train` also rejects changed metrics configuration. To intentionally change observation-only settings, use `hypergan resume RUN_DIR --config CONFIG`; its existing numerical compatibility checks still apply.

`--steps` on a new `train` sets the total schedule. Repeat that override when repeating `train`; it must still match the saved total. It does not request additional steps. `--stop-after-steps` limits only the current attempt. `--max-seconds 60` requests a cooperative wall-time stop at an update boundary; startup, a running update and final artifact writing can exceed that budget. It is not a process deadline. `hypergan resume RUN_DIR` continues using the saved configuration without needing the original project or `--steps` override. Both commands continue the saved total schedule; changing architecture, resolution, labels, losses or the schedule requires a new experiment. Partial transfer initialization is not implemented.

`--checkpoint-every N` saves after every N completed updates, with an initial checkpoint and a final checkpoint on a successful or cooperative stop. Resume inherits the interval unless overridden. A failed update may already have changed the discriminator; the failure handler does not save that partial numerical state. Resume starts from the last complete checkpoint, so work since that checkpoint may be repeated.

`inspect` exposes the latest observed update (`steps`), `last_durable_step`, `checkpoint_path`, `checkpoint_every`, `durable_event_boundary`, and `possible_lost_steps`. Only a successfully published training checkpoint establishes recoverable numerical progress. A flushed loss event alone does not. Failure status retains the error and last durable checkpoint.

On the main thread, the first **SIGINT** (Ctrl-C) or **SIGTERM** requests a graceful
stop. The controller finishes the current complete update, records its metrics,
commits a checkpoint and shuts down numerical workers. It does not start further
optional inference export or previews after observing the stop request. The manifest records
`stop_reason: SIGINT` or `SIGTERM`. A second signal forces process exit; a
30-second watchdog also forces exit if the update or shutdown cannot finish.
SIGKILL and forced exits cannot save partial updates: resume selects the last
published complete checkpoint. Native calls in embedded non-main threads retain
the host application's signal policy. A custom native extension that holds the
Python interpreter lock can delay signal handlers and the Python watchdog; a supervisor
requiring an absolute external deadline should send SIGKILL after its grace
period. POSIX worker brokers use independent sessions so terminal group signals
reach the controller without interrupting a rank halfway through an update.

Each controller checkpoint commits a **durable event prefix**. Its
`event_boundary` records run, attempt, completed step, last event sequence,
byte offset and SHA256 of `events.jsonl` through that offset. Training events
respect the configured metric cadence; disabled metrics are not synthesized.
The controller flushes and fsyncs the event file and run directory, then writes
and fsyncs checkpoint payloads and their containing directories. Finally it
atomically replaces the `latest.json` reference and synchronizes its directory.
The checkpoint manifest contains the event boundary; the run manifest mirrors
it as `durable_event_boundary`. Checkpoint notifications follow this commit and
are not required for restoring its numerical state. Projection processing is
asynchronous and can lag behind the durable event prefix.

On POSIX filesystems supporting these operations, the published reference
selects both durable payloads and their durable event prefix. Separate files are
not one filesystem transaction. Windows synchronizes file contents and replaces
the reference atomically; this implementation cannot fsync directories there,
so power-loss durability of directory entries depends on the filesystem.
A failure before reference publication leaves the previous checkpoint selected;
a failure after replacement may leave the new complete checkpoint selected even
if the run manifest still reports an earlier step. Resume reads the checkpoint
reference and validates its payload and event prefix before publishing a new
attempt. Unselected staging directories are never automatically promoted.

Missing or corrupted **committed** event bytes are an error, not an implicit
rollback that conceals data loss. Select an earlier intact checkpoint explicitly
with `--checkpoint` if needed. Newer complete events remain in the log; an
incomplete trailing row is removed before appending. Resume events record the
parent attempt/checkpoint and restored step, so abandoned updates remain
distinguishable from resumed history.

Every resume, including repeating `train`, creates a new attempt with a monotonic index and unique identifier. Repeating a completed run validates and restores it without additional training updates; it still records a new attempt and final artifacts. Existing attempt artifacts are never overwritten. `sample RUN_DIR` uses the bundle selected by the run manifest, and repeated sampling uses unique filenames. Explicit `--output` refuses an existing file. Older checkpoints remain available through `resume RUN_DIR --checkpoint PATH`; replaying one creates a new attempt without replacing earlier samples. Use `inspect` to select the checkpoint's recorded path.

The checkpoint includes generator/discriminator/auxiliary state, prior, EMA, Adam states and original learning rates, update counters, named RNG streams, global CPU Torch/Python/NumPy RNG states, and the declared data state. Resume validates the saved configuration and execution/data identity before continuing. `--config CONFIG` verifies that a supplied configuration matches; it does not override the checkpoint.

HyperGAN release identity is provenance, not a resume gate. `source.hypergan_commit` records the full Git SHA and `source.hypergan_dirty` records whether the source checkout had changes; installed wheels retain build-time provenance. Unknown source identity is recorded explicitly. The run preserves `initial_source`, while its current `source` and immutable attempt manifests identify each continuation. Checkpoints also carry their writing source.

Native and replicated checkpoints use `hypergan_checkpoint_version` (currently 1). Existing schema-1 checkpoints without that field use version 1. Supported versions can resume across HyperGAN Git SHAs and package versions. A known incompatible change to state or continuation semantics must bump the compatibility version; unknown, malformed or unsupported versions fail explicitly. This does not waive payload integrity, recipe/schedule, external component and dependency source, data, numerical runtime or fixed-topology checks. Compatibility means saved state can be restored; identical learning trajectories across changed numerical algorithms are not implied.

Built-in synthetic data is stateless apart from the trainer's RNG. `image_folder` records its content/preprocessing/class-map identity, shuffled order and cursor. Custom data must implement the documented state protocol or explicitly declare itself stateless to support recovery. Custom components must register their tensor state and obey the recovery contract; arbitrary Python caches and external services cannot be inferred from a model's weights. Unsupported recovery does not silently become a successful restore.

For process managers, `--progress-json` writes cadence-filtered JSONL progress and immediate lifecycle events to stdout, followed by a `result` event containing the final manifest. Diagnostics remain on stderr. Without that flag, routine updates go to stderr every 100 steps and the final manifest goes to stdout. Change the interval with `--progress-every N`; collected metrics are unaffected. The Python API starts no server. CLI output is bounded best-effort delivery; the event journal is authoritative. The CLI can launch the optional browser viewer.

For bounded event pages, periodic previews and acknowledged manual checkpoint requests, see [run observation](observation.md). Periodic preview retention (at most `--preview-keep` generations, 128 by default, thinning the older samples instead of dropping the beginning of the run) never deletes complete checkpoints or final attempt inference bundles. A resume inherits a stored `preview_keep` only when the manifest's `preview_keep_source` records it as explicit, so a run started under an older default is not pinned to it. A submitted save request is pending until the trainer acknowledges a durable checkpoint at a safe boundary.

The manifest and event stream are versioned for future observers. A run has one writer; concurrent resume attempts are rejected by a process lock. On restart, a partial trailing event is handled without treating it as a complete update. Checkpoints load tensor/basic-value state with `weights_only=True`, but configured Python factories still execute trusted code. Use artifacts and implementations you trust.

## Internal distributed recovery

The same public commands select native or fixed-topology replicated execution through [execution profiles](execution.md). Resume infers the saved profile; world size and accumulation must match, while operational deadlines can change. The [distributed checkpoint API](distributed-recovery.md) documents the complete rank-state format and parent-owned publication protocol.
