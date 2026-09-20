# Next-version image training plan

Accepted owner direction, 2026-09-19. This plan supplies the next image milestone
within the [resurrection plan](resurrecting-hypergan-plan-2026-09-18.md); the
[status ledger](resurrection-status.md) remains the implementation record.

## Product outcome and decisions

Make one small, proven image recipe easy to train, watch, interrupt, resume and
evaluate through HyperGAN. Its numerical foundation is **ParticleGAN MoG +
b-cap**, and its discriminator uses **pretrained features from the first version**.
The generator starts from scratch. Familiar convolutional/SAGAN-style components
provide the architecture; a conventional Gaussian-prior/vanilla-DCGAN training
recipe is not the agreed baseline.

The first supported recipe must have measured training behavior, explicit weight
and data provenance, bounded preflight and complete recovery. Keep ordinary Python
components and configurable objectives; custom combinations receive an unqualified
warning and actual incompatibilities fail. Do not market arbitrary configurations
as reliable merely because they instantiate, or promise that b-cap prevents every
possible training failure. Supported defaults earn their status through evidence.

Pretrained discriminator support is a product requirement, including correct
partial freezing, feature modes, input gradients, higher derivatives and recovery.
Faster useful convergence is a hypothesis to measure in elapsed GPU time as well
as updates. The first result should support a reproducible CIFAR-10 comparison;
competitive placement follows matching the chosen benchmark protocol.

## Source experiment and evidence

The owner identified ParticleGAN branch `feat/cifar-ae-gan-pretrained-encoder`.
Local and remote branch tips agree at
`9e9ce96c96948197e21e1171c8394e3819bb0013`. This supersedes the earlier
latent-64 residual/1,024-component direct-GAN candidate as the first port target.
Keep that earlier experiment's reports as historical evidence.

Pinned upstream sources:

- [200k findings](https://github.com/255BITS/ParticleGAN/blob/9e9ce96c96948197e21e1171c8394e3819bb0013/reports/cifar-particle-ae/sagan_gd_16k_200k/FINDINGS.md)
  and [configuration](https://github.com/255BITS/ParticleGAN/blob/9e9ce96c96948197e21e1171c8394e3819bb0013/configs/cifar_particle_ae/sagan_gd_16k_200k/sagan_gd_16k.yaml).
- [Trainer and attention implementation](https://github.com/255BITS/ParticleGAN/blob/9e9ce96c96948197e21e1171c8394e3819bb0013/experiments/train_cifar_ae_sagan.py),
  [image components](https://github.com/255BITS/ParticleGAN/blob/9e9ce96c96948197e21e1171c8394e3819bb0013/lib/image_particle_autoencoder.py)
  and [pretrained feature critic](https://github.com/255BITS/ParticleGAN/blob/9e9ce96c96948197e21e1171c8394e3819bb0013/lib/image_moonshots.py).
- [Unchanged continuation](https://github.com/255BITS/ParticleGAN/blob/9e9ce96c96948197e21e1171c8394e3819bb0013/reports/cifar-particle-ae/sagan_gd_16k_300k/FINDINGS.md)
  and [later attention-depth intervention](https://github.com/255BITS/ParticleGAN/blob/9e9ce96c96948197e21e1171c8394e3819bb0013/reports/cifar-particle-ae/sagan_gd_depth2_240k/FINDINGS.md).

Local run evidence is under
`/home/martyn/dev/ParticleGAN/runs/cifar_particle_ae/sagan_gd_16k_200k/sagan_gd_16k/`:
`config.yaml`, `metadata.json`, `metrics.jsonl`, `provenance.json`, `source.zip`
and checkpoints. Read-only inspection verified the archived trainer, image
components and feature critic match those files at the branch tip. No ParticleGAN
checkout or uncommitted files were changed; no checkpoint was executed or loaded.

| Observed result | Interpretation |
| --- | --- |
| FID50k **12.5344567 at 200k**, batch 64 | Final and best sampled checkpoint of the original run; the 200k attempt resumed the 40k scout |
| FID50k 19.6425 at 40k | An earlier point on the same trajectory |
| Unchanged continuation through 260k: best 12.6204 | More duration did not improve the saved 200k best in that interval; run was user-stopped |
| Expanded attention: best 12.2464 at 210k, final 12.9330 at 240k | A later joint G/D intervention from 200k, with new identity-initialized blocks; not the original no-fade-in recipe or a scratch-training result |

These are recorded upstream results, not HyperGAN reproduction or verified
leaderboard placement. Comparisons in the upstream report use unequal budgets
and do not isolate an attention or pretraining benefit.

## First recipe contract

| Part | Source behavior to preserve and validate |
| --- | --- |
| Data/output | CIFAR-10 RGB32; explicit source preprocessing, training augmentation and sampling order; float images in `[-1,1]` |
| Generator | Scratch deconvolution network, latent 64 to 256×4×4 then 128/64/3 channels; GroupNorm with 8 groups, ReLU, tanh output |
| Attention | SAGAN-style spatial attention at 16×16 in G and the trainable pixel D branch; active from initialization, residual coefficient 1, no phase-in, learnable gate or spectral normalization; this is not canonical SAGAN |
| Discriminator | Frozen ResNet18 `IMAGENET1K_V1` layer1–3 features, frozen BatchNorm/eval behavior, plus trainable pixel branch and feature heads; unconditioned scalar score |
| Pretrained input | Source records 64-pixel feature input and feature-state SHA256 `5de287ab28d569dfc53a5bca4a646d4416621da29e71e80859e6117c7f90b0ac`; pin the actual weight file and complete normalization/resize pipeline separately |
| Prior | Standardized MoG, 16,384 components, latent 64, fixed sigma `0.212616428732872`; retain the source initialization/calibration history rather than silently recalibrating |
| Objective | Rp-logistic, exact autograd b-cap coefficient 1 every 8 updates with lazy scaling ×8, source ParticleRegularizer/VICReg settings |
| Updates | One D and one G update, separate real/prior draws for the phases; batch 64 |
| Optimizers | Fused Adam in source; G/E LR .0003, D .00045, prior .003; G/D betas `(0,.999)`, prior `(.5,.999)`; constant learning rates; EMA .995 |
| Auxiliary encoder | Scratch encoder; reconstruction weight 1 with encoder-only gradient routing. G/prior receive no reconstruction gradient. Preserve or explicitly validate removal before claiming the same recipe |

The encoder's presence still affects construction order, RNG, optimizer ownership,
observations and checkpoint state. Detached reconstruction gradients alone do not
prove that removing it preserves the trajectory. Compare the source loop and
pinned package numerical primitives before assuming they are interchangeable.
Record any intentional deviation as a new qualified recipe/protocol.

For encoder-only reconstruction, gradients pass through G into E while G's
parameters and prior means receive none from that term; wrapping the entire
generator forward in `no_grad` would break this contract. Match supplied draws
and initial state when comparing the two loops. The source constructs models on
CPU before CUDA transfer and uses its own named RNG layout and fused Adam; the
current HyperGAN adapter has different construction/RNG/optimizer behavior.
Declare justified cross-implementation tolerances before inspecting results,
and separately require exact supported-run recovery around lazy-penalty boundaries.

GroupNorm avoids the training-mode BatchNorm issue raised for a conventional
DCGAN. This makes the candidate suitable for investigating the existing replicated
strategy, but does not establish two-GPU parity or attention/b-cap compatibility.
Pretrained parameter freezing must retain gradients with respect to generated
images, including b-cap double backward. Module modes and parameter masks remain
separate, explicit factory responsibilities.

## Implementation order and completion gates

Each row is a bounded milestone that may use several small PRs into `develop`.
Use external worktrees and subagents for independent implementation/review.

| Order | Work | Done when |
| --- | --- | --- |
| I1 | Preserve factory-defined frozen parameter masks; define pinned pretrained-weight loading/provenance; capture source recipe and protocol | A trainable head cannot accidentally unfreeze its backbone; frozen weights/buffers stay fixed, input and b-cap gradients work, optimizer ownership and full recovery are checked. Missing/wrong weights fail before a long run; no implicit downloads. Resolve source/weight terms before distributing a port |
| I2 | Port the selected G/D/prior and required encoder behavior into ordinary Python components; represent exact phase draws, augmentation, optimizer and initialization choices | Small controlled native-CUDA comparisons cover complete updates against the pinned source, then bounded image training and exact current-run stop/resume pass. Public package APIs replace imports from the upstream experiment tree. Differences are explained, not hidden |
| I3 | Add bounded PNG image grids and browser rendering, fixed-latent history and fresh samples; finish the image walkthrough | A user can prepare data, train, watch meaningful images, stop, resume and sample a saved generator in a fresh process. Previews preserve numerical state and remain useful with metrics disabled; sampling is separate from measurement |
| I4 | Add the explicit pinned FID evaluator and reproduce the source trajectory on one local GPU | Evaluation agrees on reference/sample/weight/preprocessing choices. Publish samples, quality versus updates and elapsed GPU time, peak memory, failures and complete recovery evidence. Use the 40k/200k observations as comparison points, not guaranteed scores or automatic plateau rules |
| I5 | Qualify the actual image recipe on both local GPUs | At fixed global batch and declared normalization/accumulation behavior, complete numerical/recovery checks pass; image quality, throughput and memory are measured. Exercise rank failure and resumed training; a communication diagnostic cannot close this milestone |
| I6 | Produce an externally comparable CIFAR-10 result and submit where eligible | The published configuration, checkpoint, commands, hardware/budget, pretrained-data disclosure and evaluation protocol let another user reproduce the result; external acceptance or placement is recorded only after it happens |

I1/I2 establish the image contract; image rendering work can proceed alongside
them. Qualify frozen-feature gradients and attention higher derivatives before a
long GPU run. Keep small CPU correctness CI and the installed-package workflow.
Use staged local runs with explicit wall-time/update limits and visible durable
progress. Do not launch a broad architecture sweep or seed-only repetitions as
part of this port. Any later benchmark repeat requirements get an explicit budget.

The earliest concrete implementation PR is the mixed-trainability fix and its
native/replicated recovery regressions. The next product result is the complete
pretrained-discriminator image workflow, not a standalone freezing feature.

## Evaluation and competitive target

Start with **unconditional CIFAR-10 at 32×32**, using the full training set and
the existing source experiment as the first reproduction target. Match the
recorded FID protocol: 50,000 generated samples against 50,000 training references;
`torch-fidelity` 0.3.0 compatible Inception-v3 2048 features, recorded weight file,
float32 extractor with TF32 disabled, float64 statistics, and RGB uint8 generated
by clamping then rounding `(x+1)*127.5`. Pin the complete source and weight hashes
in the implementation. Keep EMA/final-checkpoint reporting and any best-checkpoint
selection explicit and separate.

For external comparisons, select the target table's exact protocol first.
[Clean-FID](https://github.com/GaParmar/clean-fid#cleanfid-leaderboard-for-common-tasks)
provides CIFAR-10 full-data and 10%/20% data comparison tables and invites new
results. Its displayed CIFAR table uses 10,000 generated images against the test
set, so the source FID50k/train score cannot be inserted as an equivalent number.
A separate evaluator definition must distinguish these protocols. Repository
invitation is not guaranteed current submission acceptance or a ranking claim.

After full-data reproduction, **CIFAR-10 with 10% training data** is the proposed
second small benchmark for testing the usefulness of pretrained features. Fix the
subset, disclose all external pretraining data and check eligibility before
making a comparison. Keep conditional and unconditional tasks separate. Select
matching pretrained-feature baselines as well as relevant from-scratch references;
existing pretrained-feature GAN research means pretraining alone is not a novelty
claim. See [Projected GANs](https://arxiv.org/abs/2111.01007).

Report FID alongside image grids, diversity/nearest-training-example diagnostics,
quality-versus-time curves and resource cost. Training losses and color moments
do not establish image quality. Agree target scores only after measuring the
ported baseline under the chosen external protocol. The differentiation to prove
is a dependable, fast route from pretrained components to a useful generator,
with reproducible evidence and complete recovery.

## Version scope and continuity

The core runtime, viewer/metrics infrastructure and public CPU/CUDA profiles are
already implemented at HyperGAN `11b4ccc2663beef185709e3d5dcbd59b0bc593f8`.
Both post-merge required workflows passed. None of I1–I6 is claimed complete by
this planning update. The earlier explicit upstream licensing gate remains open;
this inspection did not re-audit project-wide license grants.

After the real local image and two-GPU gates, retain the version plan's separately
budgeted real two-node training/recovery, deployment and release gates. The two
local GPUs are authorized; paid compute and release publication are not part of
this planning change. Preserve historical source attribution and run evidence.
No compatibility with the research checkpoint format is required: port/reproduce
the recipe, then guarantee complete validated recovery within the current
HyperGAN implementation.

Planning evidence, copied run metadata/configuration, source hash comparisons,
documentation validation and PR integration receipts are retained under
`/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-image-plan/`.
