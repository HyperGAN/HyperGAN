# Constant learning-rate step extension — 2026-09-20

Follow-up to the [memory/startup audit](memory-leak-audit-2026-09-20.md).

The CIFAR configuration changed `training.steps` from 200,000 to 500,000. Both its saved and requested `training.lr_floor` are `1.0`, which keeps the learning-rate multiplier constant. Nevertheless, repeated `train` rejected the change with `Training configuration differs from the original run in training` because resume validation treated every schedule change as a numerical incompatibility.

## Change

Resume now accepts an increased total step target when both configurations have `lr_floor = 1.0` and every other numerical setting is unchanged. Repeated `train` continues to require identical observation settings. Explicit `resume --config` retains its existing permission to change observation settings. Decreases, annealed schedule extensions, optimizer changes, and other numerical differences remain rejected.

The same rule applies at public preflight, the controller's validation under the run lock, native checkpoint restoration, and replicated/distributed restoration. The run manifest and new checkpoints record the extended configuration, hash, and total. Existing checkpoint bytes and hashes remain unchanged; checkpoints written before the extension remain usable. A subsequent `resume` without a configuration inherits the extended target.

Learning-rate calculations and the user's optimizer settings were not changed. CLI help and recovery/execution documentation now explain this exception.

## Verification

- Native CPU regression coverage compares every restored checkpoint tensor, optimizer state, RNG state, and data state against uninterrupted constant-rate training. It covers completed and stopped runs, inherited extended targets, replay of pre-extension checkpoints, unchanged old checkpoint bytes, and rejected decreases/other changes before run mutation.
- Focused recovery, public routing, configuration and distributed identity/adapter suites: **137 passed, 29 deselected**.
- Checkpoint compatibility, lifecycle, run-state and CLI suites: **95 passed, 3 deselected**.
- Real two-rank CLI regression: **1 passed** in 19.12 seconds. A completed 3-step run extended to 6 steps exactly matched uninterrupted training, as did replay from the unchanged 3-step checkpoint. All worker processes were reaped.
- These suites ran in a temporary editable test installation. The first system-Python attempt had one isolated-import environment failure (`python -I` excludes its user-site installation); the editable installation resolved it.

Actual launcher verification from `/tmp`:

```sh
/home/martyn/dev/hypergan/training-runs/start.sh --stop-after-steps 8
```

Exited **0**. Resumed at **106008**, advanced to **106016**, wrote a full checkpoint, and stopped with `stop_after_steps`. The persisted total is **500000**, `lr_floor` remains **1.0**, and all eight recorded `optimizer/lr_scale` values are **1.0**.

Checkpoint:

```text
/home/martyn/dev/hypergan/training-runs/train-develop/checkpoints/0003-555e794eca3e49b78afb7adb05f62127-step-00106016-1ba0c556efba
```

Run configuration SHA256: `0605aa76c689f0556fd0822d3f5e928a4d3f6c8550ed52dddaced67cb377fbe7`.

Training is stopped after the bounded check. Running the normal launcher resumes the existing state toward 500,000 steps. The startup log is `/tmp/hypergan-step-extension-startup-2026-09-20.log`.
