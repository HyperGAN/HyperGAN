# Bounded replicated observation checkpoint

Date: 2026-09-19. Baseline: `bc95251155f382b3539726b718b96c8b60920cb0` on develop, following the [internal replicated adapter](core-replicated-service-2026-09-19.md) and checkpoint-policy PR [#315](https://github.com/HyperGAN/HyperGAN/pull/315). This checkpoint implements isolated CPU previews and bounded progress delivery. Public distributed train/resume, GPU/NCCL, real clusters and the browser remain separate work.

## Behavior and ownership

The replicated adapter uses the shared preview scheduler and sample reservations. At a complete boundary, rank zero captures copied EMA generator dependencies, prior state and bounded conditioning. The parent launches a separate renderer through the existing broker, with one worker and no initialized process group. The renderer reconstructs trusted configured components from a bounded weights-only snapshot and returns a bounded JSON artifact. Only the parent publishes the immutable preview generation, index and retention changes.

The first implementation deliberately permits one synchronous preview. Its deadline includes renderer startup, computation and shutdown; training snapshot capture uses the training command deadline. Numerical workers wait outside Gloo and their broker independently monitors liveness and total time. Render failures remain optional only while the training group is healthy; snapshot-command or rank failures remain fatal. When a completed or restored batch is available, final inference remains a required rank-zero command under its command deadline.

Progress callbacks must be importable Python functions. Each event runs in a fresh supervised group-free process with no persistent callback globals. JSON is bounded, delivery is sequential, and startup/callback/shutdown share one deadline. A runtime failure disables the callback for that attempt, adds a bounded durable progress observation error and emits an error event without recursive callback delivery. Native single-process callback behavior is unchanged.

The worker service's new explicit group-free mode allows exactly one worker and does not import numerical dependencies itself. Normal 2–64-rank Gloo behavior remains unchanged. Both modes retain run/attempt/sequence fences, independent coordinator-death detection, deadlines and managed-worker reaping. The parent controller and observation orchestration remain importable without Torch, ParticleGAN, NumPy or Pillow.

See the [observation guide](../docs/replicated-observation.md) for policies, limits and callback lifetime. Trusted custom code can allocate memory, access external state and create unmanaged descendants; process isolation is not a sandbox. Parent-side persistence has ordinary filesystem semantics. Coordinator death can leave unreferenced private snapshot files; these are not previews, checkpoints or resume targets.

## Validation receipt

Three subagents supplied renderer implementation, bounded progress delivery and independent whole-job acceptance, then cross-reviewed the integration. The coordinator integrated their work and owns the installed-package, PR and branch-preservation receipts.

The six new whole-job cases cover accumulated stochastic state with observations on/off; stop/resume and earlier-snapshot replay with retention/counters; collective-dependent and native-hung renderers; failing and native-hung callbacks; and actual coordinator SIGKILL during rendering followed by exact fresh-group takeover. The selected seven-case source-overlay run, including the updated strict-rejection test, passed in 124.11 seconds. Focused tests additionally cover streaming snapshot limits, custom serialization hooks that mutate copied state/RNG, blocked callback output, base-only imports, cleanup failures and preservation of primary interrupts.

Review found and fixed an exception-precedence bug: a simultaneous training-health failure could replace a callback's primary interrupt. The initial wheel at `09f7baeb` passed 429 tests; three newly added base-only regressions then reproduced that bug against that pre-fix wheel. The final source/test head is `39611e2c`. An earlier targeted snapshot test also exposed Torch's ZIP finalizer masking the original byte-limit error; the writer now retains the actionable cause. Initial results remain preserved, with no widened tolerances or weakened bounds.

The wheel built through the source distribution at source/test head `39611e2c` passed **434 installed-package tests in 465.60 seconds** outside the checkout, plus **239 base-only tests in 7.89 seconds** with Torch, ParticleGAN, NumPy and Pillow absent. Both runs used `python -I -m pytest /absolute/test/path --import-mode=importlib -q` from `/tmp`. The installed train/stop/resume walkthrough also completed five updates, five previews and 18 callback deliveries with two retained previews and no observation errors. Eleven existing numerical/lifecycle dependency files remain byte-identical to the baseline; no training mathematics or tolerance changed. PR checks, reviewed head, merge and verified branch preservation are recorded in the durable integration receipt. Durable evidence is stored outside the repository at `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-bounded-observation/`.

## Next workflow

- [x] Keep one shared lifecycle and immutable attempt identity for native and replicated runs.
- [x] Supervise persistent numerical workers and restrict canonical checkpoint publication to the parent.
- [x] Implement copied-state preview capture and bounded rendering without a training process group.
- [x] Bound progress delivery, record optional failures and keep numerical health independent.
- [ ] Finish the remaining whole-job persistence fault gates: disk failure during staging, publication failure before/after canonical selection, and failure after D before complete G/EMA. Assert observed/durable/lost-step accounting and exact fresh-group recovery.
- [ ] Expose CPU execution profiles through public `train`/`resume`, with installed headless CLI acceptance, actionable option/routing errors and documentation.
- [ ] Qualify actual two-GPU NCCL after CPU integration passes.
- [ ] Agree a concrete real two-node allocation, cost cap and cleanup plan before using Modal credit or another provider.

Upstream licensing and admission of the selected image experiment remain separate blockers to image-quality qualification and release. The optional local server and deployment targets remain later work. Older checkpoint formats and source identities need no compatibility or migration layer; full validated recovery within current supported runs remains mandatory.
