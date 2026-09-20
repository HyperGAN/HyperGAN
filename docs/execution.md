# Public execution profiles

`train CONFIG --run-dir RUN_DIR` starts a new run or resumes the latest complete checkpoint in an existing run. A new run uses the recipe's configured native device. New projects target CUDA;
`--device cpu` creates an explicit small correctness fixture. Native runs use one
GPU (including a configured `cuda:N`). To train on the two visible local GPUs:

```sh
hypergan new demo
hypergan train demo --run-dir runs/demo --profile cuda-replicated-nccl --steps 100 --no-server
hypergan resume runs/demo --no-server
```

The named replicated profiles default to two workers and one microbatch per rank.
`cuda-replicated-nccl` requires `training.device="cuda"`; each rank owns its visible
GPU index. `cpu-replicated-gloo` requires `training.device="cpu"`. `cpu-single`
selects the existing native CPU path explicitly. Incompatible devices fail;
profiles never move a recipe silently to another device.

For another fixed worker count or accumulation factor, pass a separate TOML file:

```toml
schema_version = 1
[execution]
name = "cuda-replicated-nccl"
world_size = 2
accumulation_steps = 2
```

```sh
hypergan train demo --run-dir runs/accumulated --profile execution.toml --steps 100 --stop-after-steps 10 --progress-json
hypergan resume runs/accumulated --command-timeout 120 --total-timeout 7200 --progress-json
```

The global recipe batch must divide evenly across workers and local batches must
divide evenly across accumulation steps. Replicated runs remain labelled
unqualified for arbitrary recipes; local numerical/recovery fixtures do not
establish application quality or real multi-host execution. This command launches
local supervised workers. It does not provision machines or launch cloud jobs.

Repeating `train` and explicit `resume` infer the persisted numerical execution identity when `--profile` is omitted. Repeated `train` verifies the supplied resolved configuration, applying any `--steps` override before comparison. An override must match the saved total schedule. Checkpoint and preview controls inherit the saved values when omitted. An explicit profile must
match the original world size, global/local/microbatch sizes and accumulation
algorithm. `resume --checkpoint` accepts a complete earlier generation inside the
same run; otherwise both commands pin the latest complete generation. Configuration changes
that alter numerical training, including the total learning-rate schedule, fail.
Explicit `resume --config CONFIG` can change observation configuration and
cadence under the existing metrics contract; repeated `train` requires those
settings to match too. Native identity remains the recorded recipe/device and checkpoint
runtime; no format conversion is performed.

Replicated service deadlines are mutable per attempt: `--startup-timeout`,
`--command-timeout`, `--collective-timeout`, `--total-timeout`,
`--observer-timeout` and `--preview-timeout` accept finite positive seconds.
Startup and command default to the profile's preflight timeout (60 seconds),
collectives to 15 seconds, total worker lifetime to 3600 seconds, observer calls to
5 seconds and snapshot rendering to 60 seconds. Collective timeout cannot exceed
startup or command timeout. Resume starts from these defaults (or the explicit
profile's preflight defaults), then applies supplied flags; previous attempt
policy is recorded but is not inherited. Native execution rejects service timeout
flags because it has no supervised rank service. `--max-seconds` is the separate
cooperative stop budget shared by both paths.

The CLI rejects structural profile, checkpoint selection, numerical configuration
and control conflicts before starting its optional viewer. Payload integrity and
actual numerical runtime/data/external-source compatibility are checked during locked restore before
publishing a new attempt. `--no-server` keeps viewing disabled; automatic viewing
and explicit `--server` / `--open` keep their existing local startup behavior.
Progress and final results use the bounded CLI output path; persisted run events
and manifests remain authoritative when a reader is slow or closes its pipe.

The lightweight Python entrypoints expose the same selection:

```python
from hypergan.execution import train, resume

if __name__ == "__main__":
    train("demo", "runs/demo", profile="cuda-replicated-nccl",
          stop_after_steps=10, service_policy={"command_timeout": 120})
    resume("runs/demo", service_policy={"command_timeout": 180})
```

`profile` also accepts a `Path` or raw/resolved execution-profile dictionary.
`prepare_train` and `prepare_resume` return a validated selection whose `run()`
performs training; preparation starts no workers or viewer and writes no run
files. Replicated Python callbacks must be importable module-level functions;
they execute in bounded isolated observers. The CLI's own bounded sink is an
internal trusted delivery path.
