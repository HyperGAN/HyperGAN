# Single-path logo colorization experiment

The owner requested a smaller discriminator experiment after the original run
appeared divergent. That observation is not a diagnosis. The new recipe uses:

```text
E(bw(X)) -> hard particle selection -> z = center + fixed-sigma noise
G(z) -> Xhat
D(X), D(Xhat): frozen DINOv3 -> frozen random projection -> attention/head
L2 = mean((Xhat - X)^2), updating E through frozen G and detached centers
```

There are 4,096 particles, 128 latent dimensions and 256×256 images. E/G,
encoder-only L2, no optimized KL, particle regularization, lazy b-cap, optimizer,
backend and held-out metric settings retain their previous definitions. D is
unconditional and has one feature path, with no grayscale input or pixel critic.
Real/fake evaluations and the configured gradient penalty still occur.

The new `DINOv3ProjectedDiscriminator` follows the frozen pretrained feature and
random projection principle in [Projected GAN's reference implementation](https://github.com/autonomousvision/projected-gan/blob/main/pg_modules/projector.py).
It uses the final 16×16 DINO patch map, fixed random 1×1 channel and 3×3 local
spatial mixing, then trainable SAGAN attention and a scalar head (6,145 trainable
parameters at width 64). This minimal single-map adaptation does not reproduce
Projected GAN's multiscale fusion/discriminator ensemble. Projection tensors are
saved with the model. Frozen modules retain image derivatives; math SDPA supports
b-cap double backward.

The original discriminator class remains unchanged for prior configurations.
This architecture uses a fresh config and run, rather than converting checkpoint
state. No global checkpoint compatibility version changes; source SHAs remain
provenance. DINO source/weight pins and the reviewed dataset manifest are reused.

## Validation and owner setup

Source `de11816d970e9c66423e5604582d49bea5898d52` built from a clean external
worktree through source distribution and wheel. All 65 installed Python runtime
files match. Installed fast tests: **858 passed, 185 deselected** (29.63s).
Independent review found no blockers and passed all 21 component tests.

Actual DINO weights on physical GPU 1 passed image first/second derivatives,
attention/head gradients and frozen-weight preservation across an optimizer step.
Batch eight output shape was `[8,1]`, with peak allocated memory 1,515,375,104
bytes. The actual full-manifest launcher completed eight updates, including lazy
b-cap at step eight, then resumed to ten. Complete-state restoration matched
948 tensors and 407,302 other values exactly, including optimizer, EMA, prior,
sampler and RNG state. All three configured held-out metrics completed their
512-output protocols; values are execution checks, not trained quality scores.
The live viewer served all three 768×768 PNG grids (g/x/gray), with verified
content hashes and source step. The verification train and viewer are stopped. Comparing step zero to ten also
confirmed 192 frozen backbone/projection tensors unchanged while E/G, attention,
head and particle means updated; sigma stayed fixed. A checkpoint from the old
two-path experiment also restored exactly under the new installed package:
1,050 tensors and 407,414 other values.
The old AGENTS.md required a full heavy suite before training changes merged.
The owner clarified that heavy tests are for prereleases and replaced the old
AGENTS.md in `4d143c87`; the PR preserves that owner-authored file verbatim. The full heavy run was stopped immediately: 121 passed, zero
failed, 64 selected tests unfinished, in 564.83 seconds. This is partial evidence,
not a full heavy-suite pass; no heavy suite is required for this develop merge.
Fast, focused GPU checks and GitHub checks are the merge gates.

A first isolated fast-test invocation could not import the shared pytest; an
initial heavy attempt similarly could not see user-site training dependencies
under `python -I`. The dedicated environments now explicitly expose that shared
dependency directory while retaining their own HyperGAN installation. Original
failed logs are preserved; no tests were skipped to hide failures.

Machine assets are under `~/dev/hypergan/training-runs/`: the new installed
`colorization-projected-env`, config `logos-colorization-projected-256/colorization.toml`,
and fresh owner directory `train-color-projected`. `start-color.sh` now selects the validated new experiment; the same script is
also available as `start-color-projected.sh`. The owner run is unstarted.
`start-color-original.sh` retains the original config/environment. No colorization
training was active on entry, and the original `train-color` directory was absent;
the owner's active CIFAR run was not changed.

Commands, builds, logs, derivative/state receipts and original launcher copies:
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-20-colorization-projected/`.
No owner training allocation, paid compute, release or quality claim is included.

Integration: [PR #360](https://github.com/HyperGAN/HyperGAN/pull/360).

GitHub's first manually dispatched run failed one pre-existing browser test
waiting for a cancelled metric status (33 passed, one timeout); the affected
frontend/service sources are unchanged. The normal PR run at `588a5cc4` passed
all required checks including that browser suite. The failure log remains in
evidence; its underlying race was not diagnosed. Final integration status is
recorded in the ledger.
