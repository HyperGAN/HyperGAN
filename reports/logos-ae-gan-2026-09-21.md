# RGB particle AE-GAN with DINO at 256px

The owner reported that the ResNet colorization follow-up also failed to produce
useful samples, and requested a normal ParticleGAN AE-GAN experiment to separate
the reconstruction task from the 256px model setup. This change prepares that
experiment; it does not claim a collapse fix.

`examples/logos-ae-gan-256.toml` restores the previous DINOv3 projected critic with
the nonlinear convolution head and learned RGB stem. Generator, discriminator,
prior, optimizer and training sections are identical to the earlier DINO + RGB
owner configuration. The generator remains the same latent-only 256px decoder.
No colorization checkpoint is reused.

The new RGB encoder follows ParticleGAN's `particle_ae`: layer-normalized query,
nearest center with soft straight-through derivative, and a learned bounded
offset `sigma * 3*tanh(offset/3)`. It starts with zero offset. Reconstruction is
deterministic RGB MSE, weight 1, and jointly trains encoder, generator and means.
Independent uniform-prior draws still drive the GAN. There is no KL objective.
The image routing settings are mean distance and temperature .125. Lazy b-cap
changes from 8 to the AE recipe's 4; coefficient and kappa remain 1.
See [the recipe and explicit adaptations](../docs/logos-ae-gan.md).

The graph exposes means/sigma tensors rather than an upstream prior object, so
the encoder ports the small tensor routing formula. A focused test compares
forward values and every encoder/raw-prior gradient exactly against the installed
ParticleGAN `particle_ae` with nonzero offsets. It also checks that encoding
does not advance RNG. Another test executes the example's actual graph bindings
with a small generator and verifies separate GAN versus joint reconstruction
gradient ownership; its batch has RGB only, with no grayscale key.

Validation at source commit `58dd9ed20aec335634ae0352825c4eac9e2562e5`:

- Installed-wheel fast suite: **942 passed, 186 heavy deselected in 24.41s**.
  Explicitly selected `tests/foundation` and `tests/reference` with `-m 'not heavy'`.
  An initial invocation incorrectly selected the whole `tests/` tree, including
  acceptance tests outside the default testpaths. That invocation was stopped
  and its child processes terminated; its partial output is not a passing suite.
- Physical GPU1: fresh installed CLI run completed steps 1–8; resume completed
  steps 9–16 with batch 16 at 256px. B-cap executed at 4/8/12/16, and checkpoint,
  inference export and EMA previews completed. All training is stopped.
- Preview metadata confirms RGB shape `[8,3,256,256]`, `X | X_hat` comparison
  columns and a separate random shelf. Repeated-condition diversity is removed;
  reconstruction edge/chroma and random spread/chroma diagnostics remain.
- Launcher syntax and CLI help passed. The owner run remains unstarted. No
  alternate-seed experiment or convergence trial was run.

Machine state:

- Launcher: `~/dev/hypergan/training-runs/start-ae.sh`.
- Config: `~/dev/hypergan/training-runs/logos-ae-gan-256/ae.toml`.
- Environment: `~/dev/hypergan/training-runs/logos-ae-env`.
- Fresh owner run: `~/dev/hypergan/training-runs/train-ae-dino` (not yet created).
- Existing `start-color.sh`, old environments/configs/checkpoints and owner CIFAR
  process are preserved. The data manifest and hash-pinned DINO artifacts are
  reused without changes.

Durable evidence is under
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-21-logos-ae-gan/`, including
the built wheel, build/install logs, fast-suite output, smoke/resume logs and
artifacts, exact smoke config, and `validation.json`. Wheel SHA256:
`22d90ef85c40b98ca0c8e9a636284c174f77e2dacbef85bc2667118ccf96ad14`.
The sixteen updates validate execution and recovery, not image quality.
