"""Bounded local CPU worker lifecycle for internal distributed qualification.

This is a blocking developer API, not the train CLI or a cluster scheduler.
"""
from datetime import timedelta
import math
import multiprocessing
from pathlib import Path
import tempfile
import time
import traceback


def _worker(rank, world_size, rendezvous, error_path, timeout, callback, args):
    import torch
    import torch.distributed as dist

    try:
        torch.set_num_threads(1)
        dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                                world_size=world_size, timeout=timedelta(seconds=timeout))
        callback(rank, world_size, *args)
        # Successful return requires every worker to finish its callback.
        dist.barrier()
        dist.destroy_process_group()
    except BaseException:
        try:
            Path(error_path).write_text(traceback.format_exc()[-65536:], encoding="utf-8")
        finally:
            raise


def _stop(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + 2
    for process in processes:
        process.join(max(0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
    deadline = time.monotonic() + 2
    for process in processes:
        process.join(max(0, deadline - time.monotonic()))


def launch_cpu_workers(callback, *, args=(), world_size=2, timeout=60,
                       collective_timeout=15):
    """Spawn one fixed local Gloo group; return only after every worker exits zero.

    ``callback(rank, world_size, *args)`` must be a module-level picklable function.
    Call under an ``if __name__ == '__main__'`` guard in a file, using picklable
    arguments. Workers own their training objects and write explicit artifacts;
    callback return values are ignored. Python's spawn start method is always
    used, independently of the application's multiprocessing default.

    Any nonzero exit, deadline or parent exception terminates surviving workers.
    There is no automatic retry. A caller may launch a fresh whole group to load
    its last complete distributed checkpoint. Only these direct children are
    managed; callbacks must not spawn unmanaged descendants. Abrupt parent death
    is outside this local supervisor contract.
    """
    if not callable(callback):
        raise ValueError("CPU worker callback must be callable")
    if type(world_size) is not int or not 1 <= world_size <= 64:
        raise ValueError("world_size must be an integer between 1 and 64")
    if not isinstance(args, tuple):
        raise ValueError("CPU worker args must be a tuple")
    for name, value in (("timeout", timeout), ("collective_timeout", collective_timeout)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if collective_timeout > timeout:
        raise ValueError("collective_timeout must not exceed the whole-job timeout")
    context = multiprocessing.get_context("spawn")
    processes = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="hypergan-cpu-workers-") as temporary:
        root = Path(temporary)
        rendezvous = (root / "rendezvous").as_uri()
        try:
            for rank in range(world_size):
                process = context.Process(target=_worker, name=f"hypergan-cpu-rank-{rank}",
                                          args=(rank, world_size, rendezvous,
                                                str(root / f"error-{rank}.txt"),
                                                collective_timeout, callback, args))
                process.start()
                processes.append(process)
            while True:
                failures = [(rank, p.exitcode) for rank, p in enumerate(processes)
                            if p.exitcode is not None and p.exitcode != 0]
                if failures:
                    rank, code = failures[0]
                    error = root / f"error-{rank}.txt"
                    detail = error.read_text(encoding="utf-8") if error.exists() else "No Python traceback (worker exited abruptly)."
                    raise RuntimeError(f"CPU worker rank {rank} exited with code {code}; whole group stopped.\n{detail}")
                if all(p.exitcode == 0 for p in processes):
                    return
                if time.monotonic() - started >= timeout:
                    alive = [rank for rank, p in enumerate(processes) if p.is_alive()]
                    raise TimeoutError(f"CPU job exceeded {timeout} seconds; whole group stopped (live ranks {alive})")
                time.sleep(0.02)
        finally:
            _stop(processes)
            for process in processes:
                if not process.is_alive():
                    process.close()
