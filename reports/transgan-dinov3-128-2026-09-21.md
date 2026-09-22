# TransGAN-style generator with frozen DINOv3, 2026-09-21

The new `examples/transgan-dinov3-multidepth-128.toml` pairs a standalone
`examples/networks/transgan-generator-128.hndl` with the unchanged DINOv3
multidepth discriminator. All network topology and initialization are in HNDL.
The training extra now requires HNDL >=0.6.0; validation uses the published wheel.

The generator has 61,662,491 parameters and 119 graph nodes. Its single latent
has 128 dimensions and produces 128px RGB through stages 8/16/32/64/128 with
widths 1024/1024/256/64/16. Bicubic upsampling connects the first two stages;
three pixel shuffles connect the rest. Every stage has two pre-normalized
residual attention/MLP blocks, four attention heads, parameter-free PixelNorm,
and GELU feed-forward expansion by four. Attention is global through 32px,
then local to shared 16x16 windows. Native chunk/concat performs and reverses
the window partition without mixing examples. The final tokenwise linear RGB
readout and tanh preserve the data's [-1,1] range. There is no extra noise.

HNDL 0.6 supplies learned 2D relative-position bias, independent QKV/output
bias flags and initializer schemes. Every attention layer uses relative bias;
absolute position embeddings remain at each scale. Q/K/V have no bias; output
projections and MLPs have zero-initialized biases. Xavier-uniform Q/K/V use gain
1/sqrt(2), matching the fan calculation of the official combined QKV matrix;
other linear weights use gain 1. Absolute and relative position tables use
truncated_normal(std=0.02), with the initializer's default absolute bounds.

This is a reduced-depth adaptation of
[TransGAN Table 6](https://arxiv.org/html/2102.07074v4#A2.T6), with two blocks per
stage instead of 5/4/4/4/4, latent128, tanh output, and HyperGAN's existing
training recipe. It does not reproduce the paper's dataset or optimization
setup. The [official attention implementation](https://github.com/VITA-Group/TransGAN/blob/master/models_search/ViT_custom_local544444_256_rp.py)
provides the relative-bias and combined-QKV reference.

The discriminator source, pinned DINO checkpoint/provider, MoG prior, optimizer,
loss, batch64, and seed settings match the existing SAGAN/DINOv3 run. Generator
initialization differs; this is an architecture comparison, not a seed sweep.
The backbone remains frozen while candidate gradients reach the generator.

The large generator and its integer position buffers exceed the old 256MiB
preview/evaluation snapshot cap (the exported EMA bundle is about 257MiB).
Those two artifact limits are now 512MiB. Capture, bounded writes, file checks,
hashing, deadlines and isolation remain enforced. This is an artifact limit,
not a CPU memory quota. Distributed checkpoint/bundle limits are unchanged;
this launcher uses the native single-card execution profile.

The local launcher is:

```
~/dev/hypergan/training-runs/start-transgan-dinov3-multidepth-128.sh
```

It uses the dedicated `transgan-128-env` with HNDL 0.6.0, physical GPU0 UUID
`GPU-ed080e41-3193-3755-6756-f3d46c433331`, batch64, the existing pinned logos
manifest, and a new `train-transgan-dinov3-multidepth-128` run directory.
Editable network files are in `training-runs/logos-transgan-dinov3-multidepth-128/`.
Production training was not started. The existing GPU1 run was left running.

## Validation

The generator reference suite covers batch1/3 outputs and latent gradients,
train/eval RNG preservation, sample independence, exact window order and inverse
including identity gradients, window locality and shared weights, PixelNorm,
initialization, an explicit double-precision 2D relative-bias attention oracle,
learned bias updates, state loading, deepcopy, integer buffers, and EMA.

Temporary artifacts: `/tmp/hypergan-transgan-060-fast.log`,
`/tmp/hypergan-transgan-060-snapshots.log`, `/tmp/hypergan-transgan-060-smoke.log`,
`/tmp/hypergan-transgan-060-resume.log`, `/tmp/hypergan-transgan-060-batch64-smoke/`,
and `/tmp/hypergan-transgan-060-dist/`.

Installed-wheel foundation/reference suite: **1,105 passed, 188 deselected** in
105.39 seconds. This includes 14 TransGAN tests. After increasing the snapshot
limits, the targeted capture/evaluation boundary suite passed all nine tests.

On GPU0, actual batch64 training completed eight updates, saved a checkpoint,
and resumed for eight more. All losses were finite; lazy b-cap ran at steps 8
and 16. All 109 generator parameter tensors, including all 10 relative-bias tables,
changed. All five discriminator score heads changed. Every one of the 188 frozen
DINOv3 state tensors remained bitwise unchanged after each attempt, and frozen
parameters stayed out of the discriminator optimizer. Peak allocated GPU memory
was 21.371 GiB; peak reserved memory across the attempts was 23.738 GiB.

The first attempt exposed the old preview cap at step 8. After the cap fix,
step 16 published an isolated CPU EMA preview and all six diversity metrics,
with no new observation errors. The historical step 8 error remains in the run's
record. Diversity ratios at step 16 were 0.95420 full-resolution and 0.31843 after
4x4 pooling, using 16 samples. These are execution checks, not evidence of FID,
long-run convergence, or resistance to mode collapse. No images were inspected.

The launcher passes bash syntax checking and the local TOML passes HyperGAN
validation. The local HNDL files match the checked-in sources byte for byte;
production training remains unstarted.
