# Replicated CPU execution through the shared controller

Date: 2026-09-19. Baseline: develop `d65c338cf9dfbd89ab5e3abeb1a21167fe7bc47f`, after PR #313.

This checkpoint connects the persistent CPU worker service and parent checkpoint authority to the shared run controller. It is an internal, headless CPU integration. Public `hypergan train` and `hypergan resume` remain single-process while bounded observation and the remaining public integration gates are completed.

## Ownership and recovery

The controller owns the run lock, manifest, attempt history, events, checkpoint requests and sample reservations. The replicated adapter owns a fixed supervised group; trainers, optimizers, rank RNG and sampler state stay in those workers. Only the parent publishes canonical checkpoints after successful all-rank preparation, receipt validation and a fresh supervisor health check.

The controller allocates a candidate attempt identity in memory before strict restore. Workers use that immutable identity throughout their session. The controller persists the attempt only after restore succeeds. Failed compatibility checks therefore do not create another attempt or change the selected checkpoint. Successful older-checkpoint replay republishes the selected state before another update, including when the attempt stops without updating.

Resolved numerical execution settings are stored separately from mutable service deadlines and controller policy. Changing world size, global batch or accumulation strategy remains a strict recovery mismatch; changing operation timeouts does not change the numerical contract. Native and distributed checkpoint formats remain distinct.

## Failure and observation boundaries

Worker loss, command failure and checkpoint preparation/publication failure terminate the replicated attempt. An optional manual save cannot hide a failed numerical group as an observer warning. A complete checkpoint can remain recoverable when a later operation fails; a reported publication error does not necessarily mean the latest pointer was unchanged.

Filesystem events and checkpoint request receipts reuse the shared controller protocol. Bounded callback delivery and isolated periodic preview rendering remain the next slice. Unsupported replicated callback/preview options must fail before creating a run or persisting an attempt. Readers can observe the existing events and manifest without entering the numerical processes.

Final inference artifacts, required when a completed or restored batch is available, are produced under the worker command deadline before coordinated successful shutdown and terminal success. This is distinct from the planned renderer without a process group: custom collective-dependent inference can fail the whole job. Deadlines and message/file limits do not bound arbitrary custom Python memory allocation or every filesystem operation.

## Acceptance and remaining gates

Three subagents implemented the controller and adapter and supplied independent whole-job acceptance. The coordinator reviewed the integration, kept numerical source unchanged, and owns the installed-package validation, PR and preservation receipts.

Independent whole-job fixtures cover six cases: accumulated stochastic/shuffled exact recovery with older-checkpoint zero-update replay; strict recipe/profile/data mismatch with no persisted attempt; coalesced manual saves with lost acknowledgement and fatal worker loss; actual coordinator termination while both ranks hold their GIL inside native code, followed by reaping and fresh takeover; accumulated image-folder recovery; and required inference failure followed by successful export recovery at the final durable step. The image fixture uses five tiny grayscale files to qualify data/recovery integration, not image quality.

Every terminal-status write in those fixtures checks that managed rank PIDs are gone. The coordinator stays Torch-free. Rank snapshots are compared in full with exact tensor equality, including optimizers, EMA, all recorded RNG streams, sampler state and local batch. A separate inference test verifies that custom state hooks and sampling do not mutate live registered state or consume training RNG. Foundation tests cover descriptor type mismatches, both native/distributed routing directions, pre-run control validation and fatal-error propagation.

The default and paired native workflows were also run independently with the prior installed PR #313 wheel and the new wheel, through a manual save, periodic previews, a cooperative stop and resume. Complete checkpoint state, inference state apart from attempt identity, lifecycle counters and event type/step/sequence matched exactly. This is behavioral parity, not cross-source checkpoint compatibility. The new guide's exact installed example completed in 6.13 seconds with `stopped 2` then `complete 5` and no stderr output.

One initial source-overlay acceptance run passed five cases and failed a test assertion that required a deliberate rank-one exit to be reported first as rank one. Gloo can instead report the peer's connection reset first. The corrected fixture records the injected rank-one failure independently and requires rank, prepare-operation and attempt diagnostics; lifecycle, durable-state, receipt and reaping assertions remain intact. The changed case passed its focused rerun. No production behavior or numerical tolerance was changed for this correction. Original evidence is preserved under `initial-agent-test-failure/` in the durable evidence directory.

- [x] Accumulated shuffled/stochastic full state across uninterrupted and fresh-worker resumed runs.
- [x] Strict mismatch without a new attempt, timeout-only policy changes, and older-checkpoint zero-update replay.
- [x] Shared manual-save receipts, lost acknowledgement and fatal group failure during an optional save.
- [x] Actual coordinator death, rank reaping, and fresh takeover from the last durable checkpoint.
- [x] Final inference artifacts, counters, export failure/recovery and shutdown before terminal success.
- [x] Source-distribution-built wheel and separate base-only installed suite. Required PR and post-merge CI are recorded in the integration receipt.

The source-distribution-built wheel at source/test head `f1f6c55083a018ed10e43a1b1b718ac2f02d17a4` passed **392 installed-package tests in 337.14 seconds**, including 36 new cases. A separate base-only installation passed **205 tests in 2.12 seconds**, with Torch, NumPy, ParticleGAN and Pillow absent. Both used `python -I -m pytest /path/to/checkout/tests --import-mode=importlib -q` from `/tmp`; the base-only run selected `tests/foundation`. Runtime: Python 3.12.13, torch 2.14.0+cpu, ParticleGAN 0.5.0, NumPy 2.5.3 and Pillow 12.3.0. No tests were skipped or tolerances relaxed.

Wheel SHA-256: `b9b835ed29c28d55c2e76d8021d37c9acff3f69de8833920aa0805a9eb7cecb3`. Source distribution: `d682e4f9898327f206e3417da989c9e1efaca9bdae3c3914e9317d2aa4f6fdbd`. Subsequent changes are documentation only. Eight changed Markdown files have 88 resolving local links. All three original agent commits retain matching stable patch IDs after integration.

Commands, logs, runtime and artifact identity, native parity, the guide walkthrough, PR checks, merge/tree equality and verified branch preservation are retained at `/home/martyn/dev/hypergan/resurrection-backups/2026-09-19-replicated-service/`. The final integration receipt records both post-merge workflows and the clean branch inventory.

Controller source changes intentionally invalidate strict recovery from older-source native and distributed checkpoints. Distributed identity and preflight now also hash the replicated adapter, numerical worker handler and command service. Use the original installation for old runs; this slice does not relax validation or migrate checkpoints. Twelve existing numerical, configuration, artifact and persistence modules are byte-for-byte unchanged from the baseline, including both trainers and the single-process adapter. Their hashes are recorded in `numerical-source-review.json`.

No GPU execution, paid compute, dataset download, upstream architecture copying, browser server or release is included. Actual two-GPU NCCL, a separately agreed two-node allocation, image licensing/quality and deployment remain separate gates.


## Next satisfying cutpoint

- [ ] Capture immutable preview snapshots at complete boundaries; render in a deadline-controlled process with no training process group. Keep at most one pending render and preserve existing counter/retention rules.
- [ ] Bound progress delivery so slow or hung callbacks cannot block the controller indefinitely; keep broker health monitoring independent. Compare exact final state with observers on/off.
- [ ] Complete remaining whole-job staging/publication, partial-update and renderer-failure cases. Primitive failure tests remain useful but do not substitute for integration acceptance.
- [ ] Expose the CPU profile through public train/resume only after those gates, with an installed end-to-end CLI walkthrough and explicit headless behavior.
- [ ] Then qualify actual two-GPU NCCL and prepare a separately agreed real two-node allocation; retain image licensing/quality and deployment gates.
