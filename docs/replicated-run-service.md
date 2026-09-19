# Internal replicated run service

The internal replicated adapter uses the same run controller as native single-process training. It connects fixed-size CPU/Gloo or CUDA/NCCL workers to attempts, full recovery, events, save requests and final inference artifacts. The parent process does not import Torch. This is a developer integration contract; public `hypergan train` and `hypergan resume` still use the single-process adapter.

## Run and resume

For rank-owned GPUs, use the [CUDA profile and walkthrough](replicated-cuda.md). The explicit CPU example below remains a correctness fixture.

Use an importable Python file with a guarded entry point, as required by the spawned [CPU worker service](cpu-worker-service.md). Keep recipe architecture and global batch in the recipe configuration. Use the [execution profile](execution-profiles.md) for CPU world size and accumulation, and separate service policy for startup, command, collective and total deadlines.

With the CPU training dependencies installed, save this as `replicated_demo.py` and run `python replicated_demo.py` in a new working directory:

```python
from pathlib import Path

from hypergan.config import write_default
from hypergan.replicated_execution import run_resume, run_train


if __name__ == "__main__":
    config = write_default(Path("replicated-project"), device="cpu")
    profile = {
        "schema_version": 1,
        "execution": {
            "name": "cpu-replicated-gloo",
            "world_size": 2,
            "accumulation_steps": 2,
        },
    }
    policy = {
        "startup_timeout": 60,
        "command_timeout": 60,
        "collective_timeout": 15,
        "total_timeout": 300,
    }
    stopped = run_train(
        config, "replicated-run", profile=profile,
        service_policy=policy, stop_after_steps=2,
    )
    print(stopped["status"], stopped["last_durable_step"])
    complete = run_resume("replicated-run", service_policy=policy)
    print(complete["status"], complete["last_durable_step"])
```

This prints `stopped 2` and `complete 5`; worker diagnostics may appear on stderr. Profile input can also be a raw/resolved profile dictionary or a profile TOML path. Omitting the profile during resume reconstructs the numerical settings from the run manifest. Supply service policy explicitly when retaining operational limits across attempts.

Service policy defaults to the profile's preflight timeout for startup and commands, its collective timeout for the selected backend, and 3,600 seconds total. All limits must be finite and positive; the collective timeout cannot exceed startup or command timeout. The total deadline includes startup, restore, work and idle time. `max_seconds` is a separate cooperative controller budget checked between completed updates and is not a hard timeout for an incomplete update.

The controller owns the run lock for the entire attempt, including restore, final artifacts and worker cleanup. On resume it creates an attempt identity in memory, validates the numerical profile, and restores fresh workers before writing the new attempt. A strict restore failure leaves the previous manifest, attempt inventory and canonical checkpoint selection unchanged.

The numerical identity records profile name, global batch, world size, local batch, accumulation, microbatch size and accumulation algorithm. Resume does not change the original total training schedule. Operational timeouts and checkpoint/stop policy can change independently. Checkpoints also validate actual runtime, implementation source, data identity and rank state; matching the profile alone does not prove compatibility.

Custom data and objectives without the required recovery declarations remain runnable with explicit reasons and `resume_supported=false`. Such runs do not produce full training checkpoints. Custom components remain unqualified; a successful run does not establish reproducibility of hidden Python or external state.

Older-checkpoint selection is explicit. After successful restore, the controller republishes the selected state under the new attempt before another update. Even a zero-update cooperative stop therefore leaves default resume pointing to that selection. Sample reservations remain run-wide and do not rewind when selecting older state.

## Complete boundaries and publication

Every update must return an agreed complete logical step, global metrics and checkpoint readiness from every rank. Partial D/G updates and poisoned groups cannot advance observed progress or publish a checkpoint. Workers prepare checkpoint payloads, while the parent validates the rank-zero receipt and compact peer acknowledgements, checks current group health, and publishes through its process-bound authority.

The same [events and checkpoint requests](observation.md) apply: one parent appends events, polls requests at complete boundaries, coalesces saves, records request IDs in durable metadata and acknowledges after publication. A committed request can be reconciled after a lost acknowledgement. Requests targeting an older attempt are rejected rather than applied to a new one.

This first adapter treats preparation and publication errors as fatal, including during an optional save. It does not attempt to recover an uncertain group or retry a consumed publication receipt. A later failure may leave a valid complete checkpoint. In particular, an error after pointer replacement can mean latest has changed; inspect durable state rather than assuming every failed attempt left the pointer untouched.

## Observation and completion limits

Use filesystem events and the manifest for durable progress. Periodic previews and importable progress callbacks now run through [bounded isolated observation](replicated-observation.md). The renderer has no training process group; callback failures disable further delivery for the attempt while filesystem events continue. Public distributed commands remain gated on the bounded CLI output and public profile acceptance cases.

When a completed or restored batch is available, rank zero produces required inference artifacts after the final checkpoint from a copied EMA/config/batch snapshot under the command deadline. A cooperative stop at initial step zero has no batch and can succeed without inference artifacts. Other ranks return to their idle control channels outside collectives. Terminal success requires the artifact result and coordinated successful worker exit. No successful terminal status is written while the group is still running.

The artifact command still runs in a numerical worker with a process group. A custom inference component that requires peer collectives can fail or time out the whole job. Final inference does not use the isolated preview renderer. The worker deadline bounds supervised execution; message caps and post-write file checks do not bound arbitrary custom allocations or filesystem latency.

| Internal bound | Behavior |
| --- | --- |
| 2–64 fixed ranks | Two local CPU or CUDA ranks are tested; CUDA also requires at least one visible GPU per rank |
| 40 KiB execution information | Rejects an oversized runtime/source/data description explicitly |
| 64 KiB command frame, including aggregate results | Only rank zero returns full information or a checkpoint receipt; peers return compact acknowledgements |
| Default 60-second preview / 5-second callback deadline | Separate operational policy; supervisor cleanup grace is additional |
| 1,024 final samples | Larger configured counts are rejected before creating a run; no truncation |
| 256 MiB inference bundle; 2 MiB sample JSON | Checked after writing; oversized output fails the attempt |

The [checkpoint payload bounds](distributed-recovery.md) apply separately. These limits define the current correctness implementation, not a scalable artifact transport or disk quota.

The broker independently monitors coordinator death and reaps the ranks. A fresh attempt uses a new command identity and commit authority; old workers cannot publish canonical checkpoints through the supported protocol. This is a cooperative trusted-code contract, not a sandbox preventing arbitrary custom Python from writing files. Host failure, broker failure and unmanaged descendants retain the [worker service limits](cpu-worker-service.md).

## Compatibility and scope

Controller and adapter source are part of strict recovery identity. Older-source checkpoints require their original installation. There is no implicit native/distributed conversion, source-check bypass or changed-world-size recovery.

This service covers the tested fixed-topology CPU and [local CUDA fixtures](replicated-cuda.md). It does not qualify real multi-node execution, image quality or a release. See the [checkpoint report](../reports/core-replicated-service-2026-09-19.md) and [execution ledger](../reports/resurrection-status.md) for acceptance results and the next work.
