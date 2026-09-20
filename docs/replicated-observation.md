# Bounded observation for replicated CPU runs

The internal [replicated run service](replicated-run-service.md) supports periodic EMA previews and progress callbacks while keeping the numerical workers isolated from rendering and callback execution. The [public execution commands](execution.md) use this preview service and a concrete [bounded CLI output sink](observation.md#bounded-training-command-output). Arbitrary Python callbacks retain the isolated-function contract below. Native Python callbacks and previews retain their existing behavior.

## Previews

Pass `preview_every=N` and `preview_keep=K` to internal `run_train` or `run_resume`. Zero disables previews; a resume with omitted preview settings reuses the run's stored interval and retention. Sample sequence reservations, immutable preview generations, the index and pruning use the shared [observation contract](observation.md).

After a complete update, the controller reserves a sequence and requests a snapshot from the workers. Rank zero copies EMA generator dependencies, the EMA prior and bounded conditioning from the last completed local batch. Snapshot capture preserves training RNG and operates on copied modules before invoking serialization hooks. A separate supervised process reconstructs and renders the snapshot without initializing a training process group. The isolated renderer publishes completed preview JSON and updates the index; the parent consumes its bounded receipt.

Snapshot files are limited to 256 MiB. Preview conditioning/output uses the existing 16-sample, 65,536-element and 2 MiB JSON limits; counts may be reduced under that preview contract. These are checked artifact limits, not a process memory quota.

There is at most one pending preview. Its CPU rendering and artifact publication run asynchronously; busy schedules are skipped before snapshot capture. The rank-zero snapshot file handoff still copies, serializes, fsyncs and hashes state inside the capture command, before the next update. `service_policy.preview_timeout` defaults to 60 seconds and bounds isolated rendering; capture remains subject to the training command deadline. The training broker continues monitoring numerical worker health. Cleanup adds the supervisor's documented termination grace. Native training instead hands owned CPU state directly to its preview supervisor, so snapshot storage also runs off its update thread. These contracts do not bound arbitrary custom allocations or uninterruptible filesystem calls.

A renderer exception, hung forward or attempted collective is recorded as a preview observation error if the numerical workers remain healthy. A capture command or numerical-group failure is fatal. Failed previews leave reserved sequence gaps; they never rewind sample numbering. Preview failure does not delete checkpoints or final inference bundles. Final inference export currently retains its required, rank-zero command path and is not covered by optional preview failure handling.

## Progress callbacks

Define the callback as an importable module-level Python function, then pass it as `on_event`. The worker invokes `callback(event)` and ignores its return value. Keep its module lightweight and protect the driver's entrypoint with `if __name__ == '__main__'`. Closures, lambdas and callable instances are rejected before a new run is created or a resume attempt is published.

Each event is delivered synchronously in a fresh supervised process without a training process group. Callback globals do not persist across events, and mutations do not reach the coordinator or numerical workers. Use the filesystem event log for durable progress and external effects when the callback needs persistence. Each encoded event is limited to 65,536 bytes and permits only one pending callback.

`service_policy.observer_timeout` defaults to five seconds. Startup, callback execution and shutdown share that deadline, with bounded supervisor cleanup grace. Callback stdout follows the worker service's redirection to stderr; control messages use separate sockets. Blocking output and native calls run in the callback process and can be terminated. The first runtime callback failure records a `progress` observation error and disables callbacks for the remainder of that attempt. Filesystem events continue. The error event itself is not sent back to the failing callback. Only explicit optional delivery errors receive this treatment; numerical execution and native RNG/thread-restoration failures remain fatal.

Training worker shutdown still precedes terminal status. The terminal event can then invoke its own bounded callback; that callback is reaped before the Python API returns. The observer has no persistent child between events. An abrupt coordinator death causes each independent broker to terminate and reap its managed worker, including a worker stuck in native code. Unmanaged descendants, guardian or host failure, and unkillable kernel operations remain outside that guarantee.

## Policy and recovery

Observer deadlines are operational policy, separate from numerical identity. Supply nondefault policy on each invocation; it is recorded for the attempt but not automatically reused by `run_resume`. Changing observer settings must preserve full current-run checkpoint state and strict fixed-topology recovery. No migration or compatibility support is required for checkpoints from older implementations.

This is CPU lifecycle qualification, not GPU, cluster, browser or image-quality qualification. Numeric previews remain the supported artifact; image grids and the optional local server are later work.
