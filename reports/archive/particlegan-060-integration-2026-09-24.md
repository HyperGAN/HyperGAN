# ParticleGAN 0.6.0: explicit sigma integration

The shared CIFAR/logo Python environment at
`training-runs/transgan-128-env` now has the published PyPI ParticleGAN 0.6.0
wheel installed locally in the environment. The inherited user-site 0.5.0
installation remains untouched. Installation used `--no-deps`, preserving
Torch, CUDA and the other numerical dependencies. HyperGAN's training extra
now pins `particlegan==0.6.0`.

`make_prior` passes an existing recipe's `fixed_sigma` directly to the new
constructor, avoiding all nearest-neighbor calibration. Native
`prior.args.sigma` is also accepted; declaring both forms fails early. The
legacy `sigma_rel` argument is removed before construction. For recipes without
an explicit sigma, HyperGAN invokes `calibrate_mog_sigma` on the already-drawn
read-space centers and retains the old sigma, d0 and sigma_rel semantics.
Neither path redraws the centers or changes sampling RNG consumption.

The current CIFAR config's 65,536-particle, 128-dimensional prior initializes
in **0.038 seconds** on CPU with one thread and sigma 0.212616428732872.
Its previous initialization was observed inside nearest-neighbor calibration
at repeated stack samples before a bounded 22-second diagnostic was stopped.
These measurements cover prior construction, not complete training startup.

## Checkpoint migration

Resume normally rejects dependency version/source changes. The new exception
qualifies only 0.5.0 -> 0.6.0 with the exact audited old/new source digests for
`particlegan.particle_prior` and `particlegan.recipes`. All other external
implementation and numerical-runtime checks remain strict. In particular,
loss, gradient penalty and prior regularization code must still match. The
installed 0.5.0 code and existing logo checkpoint already share the current
gradient-penalty implementation; an older checkpoint with different penalty
code does not qualify just because its version says 0.5.0.

The published wheel's migrated module hashes match the independently built
candidate used for testing. A qualified resume records a warning and restores
saved prior centers, sigma, d0/read settings, optimizers, EMA and RNG normally.
Checkpoint schemas, configuration fingerprints and the HyperGAN compatibility
version remain unchanged. Unknown source hashes, other dependency changes and
downgrades remain rejected.

## Validation

- 140 focused tests passed: 139 fast checks across configuration, prior
  construction, recovery/runtime compatibility, training numerics and
  distributed compatibility; one actual two-process objective/gradient/
  double-backward check passed separately with the published wheel.
- Before upgrading, generated complete and partial CPU training runs under
  the installed 0.5.0 package for both fixed-sigma and relative-sigma recipes.
  Continuing each partial checkpoint under 0.6.0 matched the complete 0.5.0
  reference bit-for-bit across the entire saved training state. These are
  recovery fixtures, not seed-comparison experiments.
- Restored the actual logo checkpoint at **step 11,285** in memory on GPU 1
  using the published wheel and current local config. No training update,
  checkpoint publication or run-file write was performed.
- Explicit-sigma tests prohibit calls to the calibration helper and verify
  identical centers, samples and RNG states, including sigma=0. Other tests
  cover legacy relative-noise semantics, invalid/ambiguous scale settings,
  the public resume path and rejection of unaudited changes.
- New test files pass Ruff, changed legacy Python passes fatal-error checks,
  and `git diff --check` passes.

No production training was started. The local CIFAR config separately still
has generator input shape `["B", 64]` with prior `z_dim=128`; that architecture
mismatch was reported and is not changed by this dependency integration.
