# Interval snapshot evaluation

[PR #354](https://github.com/HyperGAN/HyperGAN/pull/354) adds automatic snapshot
metrics, including FID, to native and replicated training. The existing owner
run at `http://localhost:47035/` had two manual FID definitions and no evaluation
results. Its process, run files and installed environment were left unchanged.

## Configuration and behavior

Snapshot metrics accept `trigger="interval"`, positive `every_steps`, explicit
`evaluation.device` and `on_busy="skip"`. Manual evaluation remains supported.
The public CIFAR example schedules FID50k/train every 10,000 completed steps;
its FID128 smoke remains manual. Ordinary Python metric factories, explicit
inputs and configurable data/sample protocols remain supported.

A single supervised evaluator consumes an immutable EMA snapshot. Training
continues while its worker runs. Busy scheduled evaluations are recorded as
skipped; simultaneous metrics rotate admission priority. Native storage happens
in the background. Capturing/copying state and replicated rank-zero file handoff
remain synchronous boundary costs. Choosing the training GPU can cause memory
and compute contention; device selection is explicit, with no CPU fallback.

Receipts, independent evaluation streams and charts retain the snapshot's step,
attempt, hash and numerical protocol. The viewer shows configured snapshot
metrics before any result, with manual/interval cadence, device, next step,
active state, busy skips, failures, disabled metrics and cancellation. Its public
Run response and schema expose the same schedule, including a null next step
when a metric is disabled.

Normal completion and step/time budget stops drain accepted evaluations within
bounded deadlines. Signals cancel/reap the evaluator; cancellation is distinct
from actual metric failure, including races with completed futures. An evaluator
failure respects `on_error`: fail the run, or disable that metric for the current
attempt. Infrastructure cleanup failures cannot become successful optional
no-ops. A failed replicated capture poisons the group and records failure.

Resume starts cadence strictly after the restored step. Earlier evaluations are
retained under their original attempt identities, including replay from older
snapshots. The run-lock owner reconciles abandoned result registrations and
cleans reserved snapshot transport directories. No old-format migration or
checkpoint compatibility layer was added.

## Validation and evidence

Initial local/fetched/GitHub develop was `ed530fa8`; PR #347 was merged and no
PRs were open. Work used external worktrees. Bounded agents supplied configuration,
viewer, replicated capture and independent lifecycle review; the coordinator
implemented and integrated scheduling, native capture and shared publication.

Validation included:

- 82 configuration checks; the public example resolves interval cadence 10,000.
- 36 replicated adapter/capture checks, including actual two-rank complete
  state/RNG/optimizer preservation and coordinated capture-failure propagation.
- 16 initial installed native/manual/capture checks and 21 installed scheduling,
  failure-policy, cancellation-race and actual SIGTERM/SIGKILL checks.
- **48 final installed integration/API checks**, including native and actual
  two-rank asynchronous progress, complete numerical parity, save/resume and
  older-snapshot replay, worker reaping and source-identity rejection.
- **22 final actual-browser checks** and reproducible frontend build checks.
- **One final native CUDA acceptance test**, comparing complete numerical state
  across evaluation disabled/enabled, interrupted/resumed and older-snapshot
  recovery runs, with identical repeated snapshot values.
- Actual public CLI CIFAR training on GPU 0: three updates, automatic pinned
  Inception FID128 at source step 2, result registered while training reached
  step 3. Evaluation took 22.99 seconds; the whole run took 43.61 seconds.
  This is execution plumbing evidence, not a quality or throughput claim.

The final package was built source → sdist → wheel. All **58 installed runtime
Python files** match the final source. Final identity, wheel/sdist hashes, logs,
CIFAR run/config/receipts, the broader installed CPU suite and protected GitHub
check/merge receipts are retained under:

`/home/martyn/dev/hypergan/resurrection-backups/2026-09-20-interval-evaluation/`

The final-environment first collection attempt loaded an older dependency-path
installation because pip skipped the equal-version wheel; no tests ran. Installing
the candidate explicitly with `--force-reinstall --no-deps` resolved this; the
failed collection log and successful checks are retained separately. The viewer
slice also retained its earlier stale-install and uppercase-display assertion
failures before its corrected passing checks. No failures were blanket-skipped.

Representative installed commands (run from the external worktree):

```sh
CUDA_VISIBLE_DEVICES='' /tmp/hypergan-interval-final/bin/python -I -m pytest \
  tests/foundation/test_interval_cancellation.py \
  tests/reference/test_interval_evaluation.py \
  tests/reference/test_replicated_interval_evaluation.py \
  tests/web/test_web_service.py -q
CUDA_VISIBLE_DEVICES='' /tmp/hypergan-interval-final/bin/python -I -m pytest tests/browser -q
CUDA_VISIBLE_DEVICES=GPU-ed080e41-3193-3755-6756-f3d46c433331 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/hypergan-interval-final/bin/python -I -m pytest tests/cuda/test_interval_evaluation.py -q
```

Only local physical GPU 0 was used. GPU 1 continued the owner's existing training.
This adds no new two-GPU CUDA image qualification, multi-host execution, paid
compute or release. Protected PR checks and its merge receipt establish final
integration state. Next: exercise interval scheduling in an updated installation
and a new supported run, then continue the existing image workflow plan.
