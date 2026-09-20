# Image CLI execution and first local training run

2026-09-20 execution of [the image plan](image-training-plan-2026-09-19.md).
The requested acceptance is achieved locally: an installed HyperGAN CLI passes
an actual CUDA image smoke test with complete recovery, and the longer recipe is
training on physical GPU 1. Quality reproduction, image-distributed qualification
and public distribution of the copied components remain separate gates.

## Recipe and implementation

The source is ParticleGAN `feat/cifar-ae-gan-pretrained-encoder` at
`9e9ce96c96948197e21e1171c8394e3819bb0013`. The recipe uses the scratch
SAGAN-style G and routing encoder, pretrained frozen ResNet18 discriminator
features, a standardized 16,384-component MoG (latent 64; fixed sigma
0.212616428732872), Rp-logistic and b-cap (coefficient 1, kappa 1, every eight
updates), batch 64 and source fused-Adam learning rates. Reconstruction updates
only the encoder. Independent D/G draws, initialization order, named RNG offsets,
feature modes and partial parameter masks are explicit. The copied CIFAR
components and example remain local pending the plan's source-license gate;
upstream has no root license, and owner confirmation was requested before public
distribution. No pretrained weights or dataset are redistributed.

The independent core changes are split across PRs
[#330](https://github.com/HyperGAN/HyperGAN/pull/330) (partial freezing),
[#331](https://github.com/HyperGAN/HyperGAN/pull/331) (pinned FID),
[#332](https://github.com/HyperGAN/HyperGAN/pull/332) (native numerical policies,
recovery and preflight), and
[#333](https://github.com/HyperGAN/HyperGAN/pull/333) (PNG samples and viewer).
Native preflight now uses the configured device when no distributed profile is
selected. Unsupported combinations with replicated training or accumulation fail
explicitly; this run does not qualify the image recipe on two GPUs.

## Failed gate and explicit deterministic variant

The first complete source-update comparison failed its declared tolerance after
Adam amplified tiny CUDA gradient differences. A same-port diagnostic was also
nonexact. Strict deterministic execution then correctly rejected CUDA adaptive
average-pooling backward. These failed results remain in the evidence directory;
tolerances were not widened and deterministic errors were not suppressed.

The accepted variant explicitly disables TF32 and cuDNN benchmarking and enables
strict deterministic algorithms. Its feature pooling retains the exact native
adaptive-pool forward and supplies the equivalent deterministic backward for the
nonoverlapping divisible 16/8/4-to-4 feature grids, including higher derivatives.
Other resize/pool operations remain native. Saved backend settings are enforced
for training, resume and snapshot evaluation, and observed evaluation flags are
recorded. Historical source execution enabled TF32/benchmarking: this is a named
execution deviation, not a claim to reproduce its exact trajectory or FID.

Under the same strict policy and equivalent feature-pool backward on both sides,
eight complete source/port CUDA updates passed **6,854 checks** at the original
`atol=1e-6, rtol=1e-5`; initial state and draws were exact. The comparison covers
losses, gradients, parameters, optimizer state, EMA, frozen state and RNG. The
recipe's kappa-1 cap was inactive in those eight updates; a separate kappa-0
fixture proves nonzero b-cap (0.00813716836), trainable gradients and unchanged
frozen state. That fixture does not alter the training recipe.

## Installed CLI acceptance

Source was packaged through sdist into a wheel, installed outside the checkout,
and executed with no checkout import path. The CUDA smoke completed in 174 seconds:

1. `hypergan validate` and native `preflight --runtime` against real CIFAR data,
   batch 64 and the full 16,384-component prior.
2. An uninterrupted 16-update baseline and a preview/viewer-enabled run stopped
   at update 7, resumed through update 9, then finished at 16.
3. Exact complete-state comparison: **802 tensors and 1,984 other values**.
4. Replay from the older update-7 snapshot to 16, with the same exact comparison.
5. Fresh-process `sample` produced a 64-image RGB PNG; snapshot `evaluate`
   completed with pinned Inception weights and explicit data/protocol identity.

At update 16, the unchanged kappa-1 recipe produced nonzero b-cap
**0.038590192794799805** in baseline, resumed and older-snapshot replay runs;
the exact recovery proof therefore includes an active penalty.

FID128 was **442.74881607931206** at update 16. This is a smoke result with an
undertrained generator and too few samples for a quality comparison. An
identical-image fixture separately produced raw FID -0.0001325 within its 0.001
numerical tolerance. Neither establishes the historical approximately-12 FID50k.

The core and PNG slices each passed their scoped **461 installed CPU tests**;
the PNG scope includes actual Chromium. The integrated wheel additionally passed
35 focused tests, then 30 deterministic-policy tests. Required GitHub checks must
pass at each final current-base PR head before protected merges.

## Run, observation and control

Run root: `/home/martyn/dev/hypergan/training-runs/cifar10-pretrained-20260920/`.
Configuration: `cifar10.toml`; active run: `train/`; supervisor: `run-training.py`.
The schedule remains 200,000 updates, but this first allocation stops at **40,000**
updates. It trains in 10,000-update CLI segments, evaluates EMA FID50k against all
50,000 sequential, unaugmented CIFAR-10 training references, then resumes from
the same supported-run checkpoint. Checkpoints are written every 1,000 updates,
fixed-seed EMA PNG previews every 500, and scalar metrics every update. FID uses
50,000 generated samples, seed 34002 and batch 128. Evaluation is serial with
training on GPU 1 and failures stop the supervisor without automatic retries.
The supervisor checks a six-hour elapsed budget before each training segment,
including time already spent evaluating. An in-flight evaluation has its own
one-hour timeout and may extend beyond that elapsed budget.

Follow progress:

```sh
tail -F /home/martyn/dev/hypergan/training-runs/cifar10-pretrained-20260920/workflow.log
```

Request a clean stop at a complete update boundary:

```sh
touch /home/martyn/dev/hypergan/training-runs/cifar10-pretrained-20260920/STOP
```

The first 10,000-update segment completed at 07:14:10 UTC after approximately
1,124 seconds of supervision. **EMA FID50k/train was 33.46801440699488 at 10k**,
with evaluation completing in 167.2 seconds. The supervisor successfully invoked
`hypergan resume` and training continued beyond 10k. The metric receipt is
`fid50k-step-10000.json` in the evidence directory. This is the first measured
HyperGAN score under this pinned protocol, not reproduction of the historical
200k score or an externally comparable leaderboard result.
The detached supervisor survives the coordinator turn. Its `supervisor.json`
records the current CLI phase; `train/manifest.json` records current progress.
CLI JSON results, scalar events, preview artifacts and snapshot metric protocol
receipts remain durable. A standalone localhost viewer uses port 8765 and
`viewer-session.json` outside the served run; its token is private. Standalone `hypergan project train --follow` maintains the live projection; the
projector and viewer are observation processes and do not own training state.
Authenticated Chromium verified four live chart canvases, advancing projected
sequence, and the step-3,500 RGB PNG (16 images, 128×128, 47,451 bytes) against its
indexed digest. Seven grids remained visible with metrics unselected and there
were no page errors. A live `hypergan checkpoint train --request-id
coordinator-live-check` request succeeded at update 6,389 while training continued;
`live-checkpoint-request.json` preserves its complete-boundary receipt. Screenshots and the token-free receipt are in
`live-browser-proof/` under the run root.

## Reproducibility and remaining work

Frozen environment: `/home/martyn/dev/hypergan/image-training-env/`.
Wheel source: local integration commit `29d9ec87eff9bb7b45f93b35b5a3a557cf5b9074`.
Wheel SHA256: `53717577260e00558407cebd3d4e9a83436be07ed0d9bb772fe41ab22cf2801a`.
Do not reinstall this environment during the run: checkpoint source and package
identity are deliberately strict. Runtime is Python 3.12.13, torch 2.14.0+cu130,
torchvision 0.29.0+cu130 and ParticleGAN 0.5.0. CUDA visibility is pinned to
`GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce` (physical GPU 1; logical cuda:0).
GPU 0 and unrelated desktop processes were left alone. No paid compute or release.

CIFAR bytes are hash-validated before loading and matched all 50,000 torchvision
images/labels. Local ResNet18 weights SHA256 is
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`;
feature state is `5de287ab28d569dfc53a5bca4a646d4416621da29e71e80859e6117c7f90b0ac`.
Inception weights SHA256 is
`6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`.
These are existing local files; no implicit data/weight download occurred.

Evidence directory:
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-20-image-workflow/`.
Key receipts: `cli-smoke-result.json`, `source-update-deterministic-cuda.json`,
`active-bcap-cuda.json`, `fid-smoke.json`, `training-launch.json`, the original
failed comparison/diagnostic receipts and protected PR premerge/merge receipts.
Neighboring `2026-09-20-image-core`, `2026-09-20-image-recipe` and
`2026-09-20-image-previews` directories preserve subagent builds and reviews.

Next actions: inspect the 20k/30k/40k FID50k results and image grids; preserve quality-versus-time
and diversity evidence; resolve source distribution permission; finish full-data
reproduction before external benchmark claims. I5 requires actual two-GPU image
numerical/recovery validation and is not satisfied by earlier runtime/NCCL tests.
Full reproduction, external evaluation protocol, two-host execution and release
remain later gates. There is no leaderboard or historical-FID reproduction claim.
