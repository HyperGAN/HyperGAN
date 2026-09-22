# 64px counterpart of the 128px startup testbed

This is a resolution control for checking whether the original failure also
occurs at 64px. It is not a tuning solution or a leaderboard result. The normal
training recipe is in `examples/transgan-dinov3-multidepth-64.toml`, with its
two HNDL networks in `examples/networks/`.

## Local launch

```bash
~/dev/hypergan/training-runs/start-transgan-dinov3-multidepth-64-init-v2.sh
```

The launcher uses GPU 1, the diagnostic worktree, the existing Python environment,
and the live training viewer. It starts fresh with `--no-tune`, previews every
100 steps (plus the startup preview), and checkpoints every 1000 steps.
The local configuration directory is
`~/dev/hypergan/training-runs/logos-transgan-dinov3-multidepth-64-init-v2/`;
the new run directory is
`~/dev/hypergan/training-runs/train-transgan-dinov3-multidepth-64-init-v2/`.

`training-config.toml` and `launch.sh` here record the local launch inputs. They
are installation snapshots: install them into the paths above, rather than
running them directly from this directory. Copy the two example HNDL files into
the local configuration directory as `generator.hndl` and `discriminator.hndl`.

## What changes

- G keeps its 8/16/32/64px stages and ends with a 64-channel RGB projection. The
  128px stage is omitted. The RGB weights remain Xavier initialized; bias bounds
  become ±0.125 to preserve the original fan-in initialization rule.
- D keeps all five pixel blocks, their widths, the four feature heads, and their
  weighting. Pixel attention now acts at 8px, and the last pixel map is 2px.
  The pretrained DINO input is 64px, giving a native 4×4 patch grid. Its weights
  and provider revision are unchanged and frozen.
- Dataset preprocessing produces 64×64 images with the same padding,
  interpolation, inventory, ordering, exclusions, and train/held-out split.

Batch size 64, latent dimension 128, seeds, G/D rates 0.0002, prior rate 0.002,
optimizer, objective, lazy penalty interval 8, EMA, and training horizon are
identical to `logos-transgan-dinov3-multidepth-128-init-v2`.
Changed tensor shapes and parameter construction mean this is not an identical
initial state or a matched row against the 128px leaderboard. The retained
large early G stages may limit the speedup. No speed or stability claim is made.

## Dataset provenance

The existing 128px manifest has SHA256
`e3c57bade4a0b4917960b7481c5361453d7607c211cc745212651aa62050bc27`.
The derived manifest retains all 426343 entries and all other metadata. Only
`preprocessing.height` and `preprocessing.width` change to 64. Serialize with
`json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n'` and pin
the resulting SHA256:
`3ce7e5f12faf6dbe22fc4df121aea9b6d89c7d778fc61bdb9b0d7f82a347a0d9`.
The large manifest stays outside Git. This reuses the existing source inventory;
it does not claim that every source image was revalidated during this setup.

## Validation

`compatibility-smoke.json` records eight disposable native training updates on
GPU 1 with the original batch size, seed, horizon, and hyperparameters. Losses
were finite, the step-8 lazy penalty executed, and the hash of protected
parameters and buffers stayed identical. No training checkpoint was retained.
The launcher passed `bash -n`; the unchanged training settings were checked
against the original TOML. This is a compatibility check, not evidence of
learning quality. The current frozen-feature research probe explicitly expects
128px; it must be extended before using its metrics on this testbed.
