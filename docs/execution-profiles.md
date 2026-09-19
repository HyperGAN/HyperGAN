# CPU execution profiles and preflight

An execution profile describes how to run a recipe. Keep it in a separate TOML file so changing worker count or accumulation does not rewrite the model, objective or learning rate. This slice provides structural and runtime preflight; public `train` and `resume` still use the single-process service and do not accept a profile flag.

For the current single-process execution:

```toml
schema_version = 1

[execution]
name = "cpu-single"
world_size = 1
accumulation_steps = 1

[preflight]
timeout = 60
collective_timeout = 15
```

For the internal replicated CPU strategy:

```toml
schema_version = 1

[execution]
name = "cpu-replicated-gloo"
world_size = 2
accumulation_steps = 2

[preflight]
timeout = 60
collective_timeout = 15
```

Save either as `execution.toml`, then check it against a recipe:

```sh
hypergan preflight config.toml --profile execution.toml
hypergan preflight config.toml --profile execution.toml --runtime
```

The first command validates structure and prints the resolved profile without importing torch, constructing custom factories, starting workers or reading a dataset. The runtime command constructs the configured components in bounded workers and checks actual compatibility. Runtime preflight executes trusted Python factories; structural success does not imply those factories will load or agree. Neither command trains or certifies recipe quality, image quality, GPU operation or real clusters. Full distributed lifecycle and safe whole-job takeover remain separate gates.

## Fields and numerical identity

`schema_version=1` and `[execution].name` are required. Unknown fields are errors, including misspellings and attempts to provide derived batch sizes. Omitted `world_size` defaults to one for `cpu-single` and two for `cpu-replicated-gloo`; accumulation defaults to one. `[preflight]` is optional with the defaults above. Files are UTF-8 TOML, at most 65,536 bytes.

| Input | Contract |
| --- | --- |
| `execution.name` | `cpu-single` or `cpu-replicated-gloo`; both require recipe `training.device="cpu"` |
| `execution.world_size` | Integer 1–64; single requires exactly 1, replicated requires at least 2 |
| `execution.accumulation_steps` | Positive integer; single requires exactly 1; must divide the per-rank batch evenly |
| `preflight.timeout` | Finite positive seconds for the complete preflight, including startup and construction |
| `preflight.collective_timeout` | Finite positive seconds, no greater than the complete timeout |

Booleans are not accepted as numbers. The global batch comes exclusively from the recipe's `training.batch_size`. It must divide evenly across workers. With global batch 16, world size 2 and accumulation 2, each rank processes 8 samples in microbatches of 4. Accumulation changes activation storage and evaluation order; it does not multiply the global batch or trigger a learning-rate adjustment.

The resolved result has three keys: `schema_version`, `execution` and `preflight`. Its numerical `execution` dictionary contains `name`, `world_size`, `accumulation_steps`, `global_batch_size`, `local_batch_size`, `microbatch_size` and `accumulation_algorithm`. The algorithm is `retained-local-graph-v1` at accumulation one and `detached-logit-vjp-replay-v1` above one, matching the internal replicated trainer. The single profile has its own execution name; sharing an algorithm label does not make checkpoints interchangeable.

Timeouts are operational policy, kept outside numerical identity. A timeout-only change leaves the numerical dictionary unchanged. Changing world size, accumulation, algorithm or batch changes numerical identity and requires compatibility checks. Full resume also checks runtime/source/data and all numerical state; this descriptor alone is insufficient. See [distributed numerics](distributed.md), [accumulation](accumulation.md) and [fixed-topology recovery](distributed-recovery.md).

## Python structural API and checkpoint routing

```python
from hypergan.config import load_config
from hypergan.execution_profiles import load_execution_profile

config = load_config("config.toml")
profile = load_execution_profile("execution.toml", config)
print(profile["execution"])
```

`resolve_execution_profile(values, config)` accepts the same raw input dictionary and returns a fresh JSON-compatible result. Neither function mutates the recipe or imports component factories. Derived output fields are not valid raw input. Runtime consumers must validate a resolved descriptor against the recipe before trusting it.

The pure `validate_checkpoint_kind(kind, profile)` helper accepts `hypergan-training-checkpoint` for `cpu-single`, and `hypergan-distributed-training-checkpoint` for `cpu-replicated-gloo`. It rejects `ema-inference` bundles for training resume, unknown kinds and mismatched formats. It does not open a checkpoint, validate its schema/content, restore state or convert formats. A successful standalone preflight report cannot authorize or bypass resume checks. Actual checkpoint loading remains with the native or distributed recovery implementation.

## Runtime report and limits

Install the CPU training runtime described in the [quickstart](../README.md), then add `--runtime`. Single-process preflight constructs in a disposable worker without a process group. Replicated preflight starts a real fixed-size Gloo group and constructs the existing replicated trainer. Both use supervised startup deadlines; every worker must exit successfully before preflight returns success. A failed or stalled worker causes group cleanup and a rank diagnostic or timeout error.

The JSON report has `stage="runtime"`, `runtime_checked=true` and `scope="construction-only"`. It records resolved execution settings, actual runtime and numerical source hashes, data/recovery contracts, the initialized numerical-state digest, and each rank's result. The replicated strategy reports explicit post-backward gradient averaging and its batch/buffer policy. Rank identity differences identify the affected field paths. Preflight implementation hashes appear separately under `checker`; startup timeouts remain in `profile.preflight`.

The internal Python API `hypergan.execution_preflight.preflight(config, profile, expected_identity=previous_report["identity"])` compares the complete prior identity strictly, including value types, source/runtime/data and initialized state. Call it from an importable Python file under an `if __name__ == "__main__"` guard, following the [worker supervisor contract](cpu-workers.md). A report is not a training checkpoint or a resume compatibility override.

Recovery support is reported separately from construction success. Missing custom recovery declarations produce `identity.recovery.supported=false` with reasons and warnings. This does not prohibit an otherwise constructible custom recipe. Declaring recovery support does not prove serialization or restore; those operations retain their own strict validation.

The report explicitly lists checks that were not performed: data batches and model forward I/O, optimizer updates and numerical parity, checkpoint publication/restore, and GPU/cluster execution. Known incompatible CPU tensor state, rank-local BatchNorm in the replicated strategy, or unsupported accumulation behavior fails at startup. Arbitrary custom code can still fail later during training. Constructors and identity/state hooks execute trusted Python and may have their own side effects; preflight does not sandbox them.

Worker Python and native stdout go to stderr so stdout contains one JSON result. Preflight creates temporary worker reports, then removes them; it creates no HyperGAN run, checkpoint or inference artifacts. The supervisor manages direct children, with no automatic retry. Abrupt parent death and safe run takeover remain gates for the forthcoming distributed run service.
