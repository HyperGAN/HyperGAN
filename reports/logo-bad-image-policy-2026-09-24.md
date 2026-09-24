# Recoverable training-source failures

`ColorizationData` now supports explicit `bad_image_policy = "skip"` for
shuffled training against a pinned manifest. The default remains strict.
The logo 128px example and local training-runs config enable skipping with
`max_bad_images = 100` and `max_consecutive_bad_images = 8`.

Source read/decode failures are distinguished from other exceptions. The caller
consumes worker results in sampler order, logs each new exclusion with its path
and reason, and draws subsequent usable entries until the batch is full.
Known exclusions never enter later batches or prefetch queues. No file is
deleted and changed bytes never replace their manifest identity.

Sampler schema 2 checkpoints the exclusions and failure counter. Legacy
schema-1 state loads with an empty exclusion set. Restoring retains exclusions
even after a source is repaired. The total limit, consecutive limit and
all-excluded check prevent unbounded retries. Fatal batches roll back sampler,
RNG and exclusions; warnings remain visible. Exclusions after a durable
checkpoint are rediscovered following a crash.

Resume compatibility permits only the explicit strict-to-skip policy transition
for this pinned training loader, in addition to the existing constant-LR step
extension. The manifest, models, rates and other numerical recipe settings stay
checked. Policy/limits are recorded in subsequent run/checkpoint configs;
legacy fingerprints and data identities are unchanged. Evaluation, unshuffled
passes, `ImageFolder` inventory construction and manifest preparation remain
strict. This does not add an automatic source cleanup command.

Validation:

- 132 tests passed across bad-image recovery, threaded loading, colorization,
  image data/recovery, checkpoint compatibility and configuration.
- 20 recovery/data CLI tests passed; one heavy test was deselected.
- Six distributed restore-identity configuration cases passed.
- New tests cover serial/one-thread/four-thread equivalence, missing and changed
  prefetched files, replacements, repaired-but-excluded sources, malformed
  exclusion state, failure-limit rollback and propagation of programming errors.
- A CPU training fixture enables skipping from a strict checkpoint, deletes a
  source, saves an exclusion and resumes again. Its complete final training
  state is bitwise equal to uninterrupted continuation from the same checkpoint.
- The actual local logo config passes CLI preparation against its existing run
  with skipping enabled. Preparation did not execute training or alter its run.
- New test-file Ruff, changed-code fatal Ruff checks and `git diff --check` pass.

The user had resumed the logo run; its latest durable checkpoint observed during
validation was step 5000. It was left running. The next launcher restart loads
the new implementation and explicit policy. No model or optimizer experiment,
GPU benchmark, or loader optimization was performed in this change.
