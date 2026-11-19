# Local CPU worker supervision

`hypergan.cpu_workers.launch_cpu_workers` is an internal, blocking Python API for bounded distributed correctness runs. It starts a fresh fixed group using Python's `spawn` method and CPU Gloo, then calls an importable function in every worker. It does not change `hypergan train` or `resume`, launch remote hosts, or allocate GPUs.

Put the callback in a Python file and guard the parent entrypoint:

```python
from hypergan.cpu_workers import launch_cpu_workers


def worker(rank, world_size):
    import torch
    import torch.distributed as dist

    value = torch.tensor(rank + 1)
    dist.all_reduce(value)
    assert value.item() == world_size * (world_size + 1) // 2
    if rank == 0:
        print("CPU worker collective passed")


if __name__ == "__main__":
    launch_cpu_workers(
        worker,
        world_size=2,
        timeout=60,
        collective_timeout=15,
    )
```

The callback and arguments must be picklable; local closures and interactive stdin definitions do not satisfy Python's spawn contract. The worker sets one PyTorch CPU thread and calls `callback(rank, world_size, *args)`. Callback return values are ignored. Every worker must finish the callback and a final barrier before the parent returns successfully. The API accepts 1–64 local workers; numerical qualification currently exercises two.

The optional `stdout_to_stderr=True` argument routes worker stdout to stderr before importing the numerical runtime or initializing Gloo. [Runtime preflight](execution-profiles.md) uses this to keep the parent's JSON result separate from native diagnostics and custom-constructor output. The default preserves the existing callback output behavior. Dependency-import failures are captured with the same rank diagnostics as callback failures.

For single-process preflight, `initialize_process_group=False` is permitted only with `world_size=1`. This still supervises a spawned CPU worker but omits distributed initialization and the final collective barrier, matching the process-group environment of native single-process training. The default remains a Gloo group.

`timeout` is the execution deadline, including startup. Expiry triggers cleanup; termination and kill joins can add about four seconds of grace, plus scheduling and I/O overhead. `collective_timeout` bounds process-group operations and must not exceed the whole-job timeout. Any worker's nonzero exit or an expired job deadline causes the parent to terminate remaining workers, wait briefly, then kill and reap survivors. A parent exception, including `KeyboardInterrupt`, takes the same cleanup path. The raised error identifies the failing rank and its bounded Python traceback when available; abrupt exits still report the exit code. Failed jobs are not automatically retried.

Callbacks own training state, checkpoint boundaries and artifact writes. To recover a job, launch a fresh whole group and explicitly load its last complete distributed checkpoint. A worker returning successfully is not itself a training checkpoint. The supervisor manages its direct children; callbacks must not create unmanaged descendants. Abrupt parent death remains outside this blocking callback API. The separate [persistent command service](cpu-worker-service.md) adds coordinator-death monitoring and worker reaping for supervised sessions. Remote scheduling, persisted job lifecycle, signals from a batch scheduler and elastic membership remain later integration work.

The spawn and cleanup behavior follows [Python multiprocessing](https://docs.python.org/3.12/library/multiprocessing.html); rendezvous, operation timeouts and participation requirements follow [PyTorch distributed](https://docs.pytorch.org/docs/2.14/distributed.html). Tests launch actual subprocesses, exercise a collective, deliberately crash or stall one worker, interrupt the parent and verify that children are reaped.
