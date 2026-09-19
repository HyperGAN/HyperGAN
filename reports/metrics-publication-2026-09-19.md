# Scalar metrics and numerical recovery, 2026-09-19

M1b/M2 implement configurable scalar publication in the native and internal
replicated run controller. [The metrics research](metrics-first-research-2026-09-19.md)
remains the wider design; standalone serving, event maps, custom evaluators and
startup integration are separate slices.

The `standard/v1` catalog names D/G totals and their diagnostic sum, raw and
weighted D/G adversarial values, weighted regularizer/objective contributions,
learning-rate multiplier and update duration. Configuration selects defaults,
removes individual IDs, enables individual defaults from the empty preset and
sets publication cadence. `preset = "none"` really emits no metric values.
Custom factories and snapshot modes fail explicitly until their evaluator slice
exists. [Configuration examples](../docs/configuration.md#scalar-metrics) describe
this behavior.

Publication selects completed tensor results without another objective forward
or backward. Weighted adversarial values are the actual tensor contributions,
not products reconstructed from rounded output values. Upstream gradient/prior
regularizers expose only weighted outputs; catalogs explicitly mark raw values
unavailable rather than dividing by coefficients. Lazy penalties include applied
status and effective coefficient; a zero coefficient does not imply that the
callable skipped computation. Additional objectives accept explicit stable IDs
or use a content-derived default; repeated identical terms need distinct IDs.
An explicit objective ID is part of objective identity and remains fixed for
numerical resume.

Events use schema 2 with primitive metric dictionaries, status, catalog revision,
run/stream/attempt/sequence identity and sample count. The controller remains the
only writer. Catalogs are immutable, hash-validated and readable without Torch.
Start/resume records include parent attempt, restored boundary, safe checkpoint
basename and checkpoint manifest digest. Earlier-snapshot and zero-update replay
preserve abandoned history and give future view readers enough lineage to stop
ancestor segments at the selected checkpoint.

Numerical recipe fingerprints omit observation settings, and distributed
checkpoint identity also omits them from its embedded numerical configuration.
A supported observation change starts a new catalog revision without weakening
runtime, implementation, data or execution-topology validation. Numerical source
identity includes the new mandatory scalar validation module. Both adapters
validate complete finite internal outputs even when all metric publication is
disabled. Partial D/G failures never emit a complete metric row. CLI progress
prints available D/G values and falls back to the step when those are disabled.

## Validation

The source-validation environments are isolated from existing installations:
`/tmp/hypergan-metrics-config-verify` uses the existing CPU dependencies and
`/tmp/hypergan-metrics-config-cuda-verify` uses the pinned local CUDA dependencies.
Neither dependency environment was modified. Final package/CI gates belong to
the coordinator's integrated PR; these are focused source-validation results.

- 19 new torch-free configuration/catalog tests passed, including empty/default/
  per-ID selection, strict errors, finite values, immutable hash validation and
  stable objective names.
- The final focused CPU command covered all foundation tests plus new native and
  replicated metrics tests, native recovery and distributed numerical training:
  **307 passed**, with one environment-only package-inventory failure caused by
  generated `src/hypergan.egg-info` shadowing installed metadata. Removing that
  generated build artifact made both inventory tests pass. Earlier focused runs
  passed 53 numerical/configuration tests and 21 completion-validation tests.
- Actual CUDA acceptance passed **3 tests in 37.61 seconds**: native metric
  parity/recovery, complete two-GPU accumulated metric parity/recovery and the
  existing complete two-GPU numerical reference comparison. After adding the
  metrics module to source identity, the native/two-GPU metrics recovery gates
  passed again: **2 tests in 33.60 seconds**.
- New recovery fixtures compare every saved tensor/state item exactly across
  metrics on/off, cadence, changed observation config, earlier checkpoints and
  zero-update replay. They validate loss decomposition and lazy application;
  malformed completion values still fail with publication disabled.
- `git diff --check` passed. Focused final logs are
  `/tmp/hypergan-metrics-config-cpu-final.log` and
  `/tmp/hypergan-metrics-config-cuda-final.log`; the coordinator preserves durable
  integration evidence and records the PR/merge in the status ledger.

The first attempted CUDA source environments exposed missing transitive site
paths (pytest, then CUDA Torch, then ParticleGAN); no hardware test was skipped.
The resolved environment uses the installed ParticleGAN copy from the existing
CUDA validation environment and Torch 2.14.0+cu130 from the local ParticleGAN
environment. Both owner-authorized local GPUs were used. No paid allocation,
release, dataset download or upstream source copy occurred.
