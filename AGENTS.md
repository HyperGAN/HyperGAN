# HyperGAN resurrection workflow

Read `reports/resurrection-status.md` first, then the linked plan. Verify its state against Git and GitHub before continuing; do not repeat the completed audit.

The next release integrates on `develop`. Use small PRs targeting develop and separate worktrees outside this repository. The coordinator may delegate bounded work to subagents, review and merge passing PRs, and retire historical branches after verified preservation. Do not publish a release or launch paid compute as part of the foundation milestone.

Recipe configuration must remain flexible: ordinary Python components, explicit I/O, configurable objectives and regularizers. Warn on unqualified configurations and fail on actual incompatibilities. Never hide unsupported behavior behind successful no-ops or blanket test skips.

Record the HyperGAN release Git SHA and dirty state as run/attempt/checkpoint provenance. Preserve backward resume compatibility across HyperGAN releases by default; do not reject solely because HyperGAN source hashes or package versions changed. Use an explicit checkpoint compatibility version for known incompatible state or continuation changes, with actionable rejection. Keep configuration, external component/dependency, data, runtime/topology and payload-integrity validation, complete save/resume, and recovery from earlier snapshots.

Update the status ledger at milestone boundaries and before a handoff/compaction. Record commands/results, PRs, blockers and the next concrete action. Preserve historical attribution and archive evidence before deleting legacy code or branches.

GPU execution is the product default. New projects target CUDA; CPU use is explicit for small correctness fixtures. The owner authorizes this machine's two local GPUs for validation. Keep CPU CI, qualify CUDA save/resume and full two-GPU numerical/recovery behavior, then real multi-host execution. A successful NCCL diagnostic alone does not qualify distributed GAN training. Paid compute still requires a concrete agreed allocation.
