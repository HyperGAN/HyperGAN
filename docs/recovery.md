# Recover CPU training runs

The CPU reference loop has separate **training checkpoints** and **EMA inference bundles**. A training checkpoint restores the numerical run. An inference bundle loads the generator and prior for sampling; it cannot resume training. GPU and cluster recovery are still qualification gates.

Create a project and stop after two updates without changing its five-update learning-rate schedule:

```sh
hypergan new demo
hypergan train demo --run-dir runs/demo --checkpoint-every 1 --stop-after-steps 2
hypergan inspect runs/demo
hypergan resume runs/demo
hypergan sample runs/demo --count 16
```

`--steps` on a new `train` sets the total schedule. `--stop-after-steps` limits only the current attempt. `--max-seconds 60` requests a cooperative wall-time stop at an update boundary; startup, a running update and final artifact writing can exceed that budget. It is not a process deadline. Resume continues the saved total schedule; changing architecture, resolution, labels, losses or the schedule requires a new experiment. Partial transfer initialization is not implemented.

`--checkpoint-every N` saves after every N completed updates, with an initial checkpoint and a final checkpoint on a successful or cooperative stop. Resume inherits the interval unless overridden. A failed update may already have changed the discriminator; the failure handler does not save that partial numerical state. Resume starts from the last complete checkpoint, so work since that checkpoint may be repeated.

`inspect` exposes the latest observed update (`steps`), `last_durable_step`, `checkpoint_path`, `checkpoint_every`, and `possible_lost_steps`. Only a successfully published training checkpoint establishes recoverable numerical progress. A flushed loss event alone does not. Failure status retains the error and last durable checkpoint.

Every resume creates a new attempt with a monotonic index and unique identifier. Existing attempt artifacts are never overwritten. `sample RUN_DIR` uses the bundle selected by the run manifest, and repeated sampling uses unique filenames. Explicit `--output` refuses an existing file. Older checkpoints remain available through `resume RUN_DIR --checkpoint PATH`; replaying one creates a new attempt without replacing earlier samples. Use `inspect` to select the checkpoint's recorded path.

The checkpoint includes generator/discriminator/auxiliary state, prior, EMA, Adam states and original learning rates, update counters, named RNG streams, global CPU Torch/Python/NumPy RNG states, and the declared data state. Resume validates the saved configuration and execution/data identity before continuing. `--config CONFIG` verifies that a supplied configuration matches; it does not override the checkpoint.

Implementation source is part of that strict identity. The [shared lifecycle extraction](run-lifecycle.md) changes source hashes for both native and internal distributed checkpoints; use the original installation to continue checkpoints created before that change. Cross-version checkpoint migration is not implemented.

Built-in synthetic data is stateless apart from the trainer's RNG. `image_folder` records its content/preprocessing/class-map identity, shuffled order and cursor. Custom data must implement the documented state protocol or explicitly declare itself stateless to support recovery. Custom components must register their tensor state and obey the recovery contract; arbitrary Python caches and external services cannot be inferred from a model's weights. Unsupported recovery does not silently become a successful restore.

For process managers, `--progress-json` writes flushed JSONL events to stdout, followed by a `result` event containing the final manifest. Diagnostics remain on stderr. Without that flag, updates go to stderr and the final manifest goes to stdout. The Python API starts no server, and the optional browser viewer remains unimplemented.

For bounded event pages, periodic previews and acknowledged manual checkpoint requests, see [run observation](observation.md). Periodic preview retention never deletes complete checkpoints or final attempt inference bundles. A submitted save request is pending until the trainer acknowledges a durable checkpoint at a safe boundary.

The manifest and event stream are versioned for future observers. A run has one writer; concurrent resume attempts are rejected by a process lock. On restart, a partial trailing event is handled without treating it as a complete update. Checkpoints load tensor/basic-value state with `weights_only=True`, but configured Python factories still execute trusted code. Use artifacts and implementations you trust.

## Internal distributed recovery

The public commands above remain single-process. Developers can exercise separate [fixed-topology CPU checkpoint APIs](distributed-recovery.md) with the replicated trainer and bounded worker supervisor. Distributed CLI lifecycle and observer integration remain planned.
