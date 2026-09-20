# Image training: the next five steps — 2026-09-20

Agreed direction after the first successful CIFAR run: turn the result into a
reproducible reference that another user can obtain through HyperGAN. This
document sequences the next work within the [image plan](image-training-plan-2026-09-19.md)
and [version plan](resurrecting-hypergan-plan-2026-09-18.md). The
[status ledger](resurrection-status.md) records completion; the
[user feedback](feedback-2026-09-20.md) captures the current hands-on findings.

## Starting point

PR [#334](https://github.com/HyperGAN/HyperGAN/pull/334) merged the public recipe
into develop at `261ff9f3d34aa74475c57b79fadb53554680c84c`. The installed CLI
workflow, source comparisons, current-run recovery and PNG/browser proofs are
recorded in the [execution report](image-training-execution-2026-09-20.md).
I1–I3 have their main implementation and acceptance evidence; I4 remains the
immediate image-quality milestone.

The completed bounded allocation reached 40,000 updates in 5,152.71 seconds
including four evaluations. EMA FID50k/train improved from 33.47 at 10k to
19.38 at 40k. The historical ParticleGAN run recorded 19.64 at 40k and 12.53
at 200k. These are comparison points, not guaranteed future scores. HyperGAN
uses the explicitly documented deterministic execution variant.

The owner is now testing resume and the UI. Inspect live processes and the run
manifest before taking control; the recorded 40k allocation is not a claim that
the run is still stopped or still at that step. Do not start a second writer or
interrupt an owner-started run.

## 1. Continue the reference recipe through 200k on GPU 1

Continue the same formulation, data and evaluation protocol in bounded segments,
with checkpoints, image samples and periodic FID. Keep the existing run's frozen
environment intact. Establish whether the port sustains the promising 40k
trajectory through the historical 200k comparison point.

Done when: the complete trajectory, final result, elapsed cost and any failures
are recorded with checkpoint and evaluation provenance. Report the observed
result even if it differs from the historical score. This step does not include
an architecture sweep or seed-only repetitions.

## 2. Publish the reference evidence while training continues

Prepare FID versus updates and elapsed time, fixed-latent progress grids, fresh
samples, diversity and nearest-training-image diagnostics, hardware and memory
measurements, and exact reproduction commands. Include the existing complete
recovery evidence and distinguish final from best-checkpoint selection.

Done when: another reader can assess image usefulness, coverage, resource cost
and reproducibility from the report and associated artifacts. The current run
measures useful convergence speed; isolating the contribution of pretraining
requires a separately scoped matched comparison.

## 3. Make the fresh-user workflow dependable

Close the gap between a documented CLI and the cached data, weights and local
supervisor script used in the reference run. Add explicit checksummed asset
preparation, convenient recipe creation, and a supported bounded
train/evaluate/resume workflow. Retain ordinary Python configurability and a
measured default recipe.

Address the [feedback report](feedback-2026-09-20.md) in bounded implementation
PRs: console cadence and its UI control; the tensor-sample issue (image display
works); termination behavior; configurable binding/authentication; a server that
survives training completion; durable checkpoint/metric consistency; and FID as
a step-ordered evaluation metric. Review termination and checkpoint/event
publication together. Recorded design proposals still need implementation review.

Done when: a fresh installation can follow the published journey without
maintainer-written glue, inspect results after training ends and recover with a
clear relationship between saved state and metrics. Exercise the actual CLI/UI
flow and failure behavior, not just isolated component tests.

## 4. Qualify the actual image recipe on two GPUs

The generic distributed runtime already has evidence, but this recipe currently
rejects replicated execution. Implement and validate its independent D/G draws,
frozen features, encoder routing, optimizer policy and b-cap across ranks.
Compare complete updates at fixed global batch and declared tolerances, then
exercise rank failure and full recovery before measuring image learning,
throughput and memory.

Done when: the actual recipe satisfies I5 with numerical, recovery and measured
learning evidence. Existing NCCL/runtime tests alone do not qualify it. GPU 1
remains the reference-run device; coordinate both local GPUs for this later
milestone without disrupting the owner's experiments.

## 5. Produce an externally comparable benchmark result

Verify an eligible comparison target and its current protocol before choosing
the evaluator. Evaluate saved checkpoints using the required reference split,
sample count, preprocessing and checkpoint-selection policy. Disclose external
pretraining and compute, publish reproducible artifacts and submit where eligible.
Keep source-protocol FID50k/train explicitly separate from external scores.

Done when: the external result is reproducible and any claimed acceptance or
placement has actually occurred. After full-data CIFAR qualification, the proposed
next small benchmark is 10%-data CIFAR with a fixed subset and explicit external
pretraining disclosure. Scope repeat requirements and compute before running them.

## Execution order and handoff

Continue step 1 alongside evidence work in step 2 and usability work in step 3.
The immediate engineering entry point is the owner's feedback, especially the
connected termination, checkpoint/event durability and server-lifecycle issues.
Preserve the reference recipe while fixing its surrounding workflow. Step 4 is
the next substantial distributed-runtime milestone; benchmark-protocol research
can proceed independently before step 5's measured submission.

Use small reviewed PRs into develop and external worktrees. Delegate independent
implementation/review where useful. Keep current-run save/resume and recovery
from earlier snapshots complete; no cross-version checkpoint compatibility work
is required. Do not reinstall the frozen environment solely to test new changes.

Real two-host recovery, native/portable generator deployment and release
qualification remain subsequent version gates. Paid compute needs a concrete
agreed allocation. This planning commit launches no training, changes no runtime
behavior and publishes no release.
