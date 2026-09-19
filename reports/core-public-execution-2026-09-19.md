# Public local execution and bounded output

[PR #327](https://github.com/HyperGAN/HyperGAN/pull/327) exposes the existing
supervised local trainer through public `train` and `resume` commands. [PR
#328](https://github.com/HyperGAN/HyperGAN/pull/328) supplies bounded output,
including cleanup when the consumer is slow, closed or unread. The [execution
guide](../docs/execution.md) documents the usable interface.

New projects continue to target native CUDA. A named `cuda-replicated-nccl`
profile selects two visible GPUs; `cpu-single` and `cpu-replicated-gloo` provide
explicit CPU fixtures. Separate TOML profiles select fixed world size and
accumulation. Recipe components, explicit I/O and objectives stay in recipe
configuration. Custom combinations retain their unqualified warnings and actual
incompatibilities fail.

```sh
hypergan new demo
hypergan train demo --run-dir runs/demo --profile cuda-replicated-nccl --stop-after-steps 2 --no-server
hypergan resume runs/demo --command-timeout 120 --no-server
```

Resume infers persisted numerical identity even after the original profile file
is removed. Explicit profile changes must match the saved topology, batch and
accumulation algorithm. Operational deadlines can change; omitted deadlines use
fresh defaults rather than inheriting the previous attempt's policy. The
lightweight preparation API rejects configuration, profile, control and selected
checkpoint metadata conflicts before viewer startup. Full payload/runtime/source
and data validation remains inside locked restore, before a new attempt is
published. Current-run earlier and zero-update recovery are supported; no older
implementation migration or checkpoint conversion was added.

The training CLI encloses preparation, warnings, viewer startup, execution,
errors and final result in the [bounded output transport](core-cli-output-2026-09-19.md).
Its exact internal sink bypasses arbitrary callback-worker startup; replicated
Python callbacks retain the isolated importable-function contract. Native Python
callbacks retain their direct, RNG-protected execution contract. Each stream
has sixteen queued lines per stage and a 64 KiB line bound. Backpressure can drop
records, including the final result; durable events and the run manifest remain
authoritative. A large result emits an explicit `output_omitted` record pointing
to its manifest. Normal results use one complete compact JSON record.

Cross-review found and fixed two output defects before integration: pretty JSON
could be partially evicted line by line, and buffered cached Python/C stdio could
flush into restored descriptors and hang interpreter exit. Each result is now queued as one complete JSON record; cached Python streams and owning libc/UCRT stdout/stderr flush
while drains are active. Exception paths restore descriptors and reap drains.
This manages standard descriptors and their owning runtime, not arbitrary custom
streams, separate CRTs or externally held native stream locks. A separate review
fixed malformed saved configuration handling so selected checkpoint steps are
validated against the resolved, hash-checked resume schedule.

## Installed acceptance

Three subagents implemented and cross-reviewed transport, routing and independent
acceptance in external worktrees; the coordinator reviewed integration and ran
broader installed regressions. Source distributions were rebuilt into wheels and
installed into separate CPU/CUDA environments. Commands ran outside the checkout
with `python -I -m pytest ... --import-mode=importlib` or the actual
`python -I -m hypergan` entrypoint.

- The integrated source at `70d9e92e` passed **641 installed tests in 699.01
  seconds**, covering foundation, numerical reference, reducers and viewer API.
  This broader run preceded the buffered-output and malformed-metadata fixes.
- The final output slice passed **358 installed lightweight tests in 18.01
  seconds**, including twelve real transport/shutdown regressions. All nine
  Linux/macOS/Windows Python 3.10–3.12 lightweight CI jobs passed on that source.
- The coordinator reinstalled the corrected public wheel at `3a519111` and ran
  the entire foundation suite, existing native CLI tests and viewer API: **402
  tests passed in 53.54 seconds**.
- The corrected public source at `3a519111` passed **38 installed CPU tests in
  91.736 seconds**, without failures or skips. These cover preparation, output,
  full stochastic shuffled-data accumulated recovery, profile-file removal,
  changed deadlines, earlier zero-update selection, immutable artifacts, public
  sampling/events, early rejection and unread/closed stdout and stderr. Complete
  rank state and worker cleanup are checked.
- Actual two-GPU stochastic accumulated observation/recovery and earlier-snapshot
  replay passed before the final output correction. On the corrected source,
  public CUDA/NCCL train/stop/resume with required viewing matched the headless
  result across **144 tensors and 2,740 state values**. Authenticated API reads
  observed running steps one and three, current-attempt training events and
  bounded projection pages while the CLI remained active. Server/projector
  cleanup and credential removal passed.
- The corrected installed default path, `new` then native CUDA train/stop/resume
  without a profile, matched uninterrupted training across **72 tensors and
  1,370 state values**. The earlier CPU viewer proof also compared complete
  two-rank state exactly.

Runtime: Python 3.12.13, ParticleGAN 0.5.0, Torch 2.14.0+cpu and 2.14.0+cu130.
Both owner-authorized RTX A6000 GPUs were used; unrelated jobs were left running.
The full coordinator run initially lacked Pillow; the isolated environment was
corrected and the failed collection log was preserved. Its successful run also
recorded Wasmtime finalizer diagnostics after pytest completed; these are
retained rather than described as a clean shutdown proof. Follow-up `fa675695`
explicitly closes the cached WASM module and engine before dependency teardown.
Its rebuilt installed wheel passed **44 reducer/viewer tests in 9.54 seconds**
with clean process exit, including a deterministic dependency-teardown
subprocess regression. This changes observer resource lifetime, not reducer
math or numerical checkpoint state.

The baseline macOS viewer job printed twenty passing tests then stalled during
process exit. Its cancelled log and successful rerun are preserved. [PR
#326](https://github.com/HyperGAN/HyperGAN/pull/326), merged at `b78ec0a4`, adds a
ten-minute viewer-job bound and test diagnostics; it does not claim to identify
or fix the original exit cause. Required CI and any follow-up results are
recorded in the durable integration receipt.

Evidence is under
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-public-execution/`.
`final-public-acceptance.json` records the exact tested source tree, commands,
runtimes and sdist/wheel hashes; `bounded-output/` retains separate transport
artifacts and tests; the coordinator's logs preserve broad and final focused
checks. `pr-*-premerge.json`, `pr-*-merged.json` and `integration-current.json`
record exact-head required checks and final integration identities. Protected
merges use no admin bypass.

## Next implementation boundary

Public local profile routing is implemented. This synthetic numerical/recovery
milestone does not qualify image quality, a real image workload, throughput or
actual multi-host training. Next resolve the selected upstream image experiment's
explicit licensing and freeze its architecture, dataset/preprocessing,
augmentation, evaluation and weight provenance; then qualify an actual small
image recipe on native and two-GPU execution. A real two-host allocation still
needs a concrete separate agreement. No paid compute, dataset download, upstream
architecture copy or release occurred in this slice.
