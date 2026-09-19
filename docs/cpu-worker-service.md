# Internal persistent CPU worker commands

The [local CUDA/NCCL extension](replicated-cuda.md) now applies this contract to rank-owned GPUs. CPU-specific examples and original qualification evidence below remain explicit correctness fixtures; consult the [execution ledger](../reports/resurrection-status.md) for the current combined scope.

`hypergan.cpu_worker_service.CPUWorkerService` keeps one fixed CPU Gloo group alive across sequential commands. It is an internal prerequisite for the distributed run service. It does not connect profiles to `train`/`resume`, own a run lock, publish checkpoints, schedule previews or launch remote hosts.

Put the factory and handler at module scope in an importable Python file and guard the coordinator entrypoint:

```python
from hypergan.cpu_worker_service import CPUWorkerService


def factory(rank, world_size):
    # Construct the trainer or other owned state inside this rank.
    return {"rank": rank, "count": 0}


def handle(state, operation, payload):
    if operation != "advance":
        raise ValueError("Unsupported operation")
    state["count"] += payload["count"]
    return {"rank": state["rank"], "count": state["count"]}


if __name__ == "__main__":
    with CPUWorkerService(
        factory, handle, run_id="run-one", attempt_id="attempt-one",
        world_size=2, startup_timeout=60, command_timeout=30,
        collective_timeout=15, total_timeout=300,
    ) as service:
        result = service.command("advance", {"count": 1})
        service.assert_healthy()
        print(result)
```

The factory receives `(rank, world_size, *args)` and returns state that stays in its worker. The handler receives `(state, operation, payload)`. The result envelope contains `run_id`, `attempt_id`, `sequence`, `operation` and rank-ordered `results`. `start()` is available for callers that need explicit lifetime management; `close()` requests coordinated shutdown, while `abort()` stops the group without another numerical operation. Both reap through the broker. One sequential caller owns each instance; concurrent calls are unsupported.

## Ordering and bounds

- The default group has 2–64 ranks; tests qualify the two-rank CPU case. All ranks initialize one CPU thread and a fixed Gloo process group. GPU/NCCL, elastic membership and retries are not provided.
- Run and attempt IDs are fixed at construction: 1–128 ASCII letters/digits, underscores or hyphens, starting with a letter/digit. Commands have 1–64 characters, starting with an ASCII letter. Internal shutdown/health operations are reserved.
- Command sequences start at one and increase strictly. Every rank checks run/attempt/sequence and agrees the full command digest, including operation and payload, before invoking its handler. A stale, duplicate or conflicting command fails the group. These are operation fences, not proofs that arbitrary handlers implement the same mathematics.
- Workers wait on their command sockets outside Gloo while idle. Slow coordinator work does not itself use the collective timeout. The broker continues checking worker exits, coordinator death and the total deadline during idle periods.
- `startup_timeout`, `command_timeout`, `collective_timeout` and `total_timeout` are finite positive seconds. Startup and total clocks begin in the coordinator before process launch. Total time includes idle periods. A deadline stops the ranks; terminate/kill cleanup can add up to four seconds of join grace, plus scheduling and filesystem overhead. Parent response waits allow cleanup grace.
- Control and results use nonblocking, length-framed JSON sockets, separate from worker stdout. Each encoded frame is at most 65,536 bytes, including its envelope. The aggregate rank-result envelope has the same limit. Values must be JSON scalars/lists/string-keyed objects; nonfinite numbers, tuples, bytes and tensors are rejected. Use artifact references for large results. These are wire bounds, not limits on memory allocated by arbitrary trusted code or JSON serialization before the size check.

For isolated observers, `initialize_process_group=False` requires `world_size=1`. This mode imports no numerical runtime itself and performs no Gloo initialization, command agreement or shutdown barrier. Its factory and handler may import their own dependencies. Identity fences, bounded commands, deadlines and broker ownership still apply. This is a process-isolation primitive, not a single-rank distributed numerical qualification.

Worker Python/native stdout is routed to stderr before numerical imports and callback bootstrap. Do not use stdout as a control channel. Arbitrary top-level code re-executed by Python's `spawn` bootstrap is outside this redirection and must follow the guarded-entrypoint contract.

## Ownership, coordinator death and health

A separate non-daemon broker owns the ranks. It monitors the coordinator's actual multiprocessing process sentinel, not a lock file, PID polling or a thread inside a training worker. Rank sockets and bootstrap data are passed explicitly using `spawn`; ranks do not inherit a coordinator-liveness writer that would keep the sentinel alive accidentally. If the coordinator dies while a rank is inside a long native operation or holding its Python GIL, the broker independently terminates, kills if needed, and reaps its ranks. Rank failure or nonzero exit also ends the group, including failures while the caller is idle.

`assert_healthy()` queries the broker synchronously. It detects failures already observed at that check; it is not an atomic guarantee that every rank will remain alive during a subsequent filesystem commit. `sequence` and `next_sequence` are read-only properties. For checkpoint preparation, capture `next_sequence`, include it in the application payload, and require the returned envelope's sequence to match before using a receipt. Only the separate checkpoint authority can decide publication under the run lock. A successful command alone is not a checkpoint or a permission to bypass lineage checks.

The parent never kills the broker on an ordinary response timeout: it closes its channel and waits for broker cleanup. If the broker has not exited within that cleanup wait, the error says so and leaves the owner alive to reap ranks. Cleanup failures attach evidence to the primary operation exception. Errors include run, attempt, sequence and operation, plus rank and bounded traceback when available. `broker_pid` and `worker_pids` expose process identities for diagnostics.

Factory/handler/arguments are trusted picklable bootstrap inputs. They are serialized once in the coordinator, forwarded as opaque bytes by the broker, and deserialized only inside numerical workers. Arbitrary parent-side serialization and top-level main-module code executed before the broker monitor starts cannot be made interruptible by this service; do not perform blocking work there. No ranks are launched until broker bootstrap completes. The service does not sandbox callbacks or manage descendants that callbacks create. Broker/guardian failure, kernel processes that cannot be killed, host failure, and safe canonical checkpoint takeover are outside this module's guarantee. After abrupt coordinator death, the exited broker is adopted and reaped by the operating system; test containers whose PID 1 does not reap children need a subreaper harness. The broker itself reaps the managed ranks before exiting.

The [replicated run adapter](replicated-run-service.md) now connects these commands and parent checkpoint publication to the shared controller. It validates complete numerical boundaries in addition to the command protocol. The same broker supports [bounded observer delivery and isolated preview rendering](replicated-observation.md); whole-job fault acceptance and public distributed commands remain separate gates. The [blocking CPU worker supervisor](cpu-workers.md) remains available for finite callbacks.
