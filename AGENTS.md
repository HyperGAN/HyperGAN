# HyperGAN resurrection workflow

Read `reports/resurrection-status.md` first, then the linked plan. Verify its state against Git and GitHub before continuing; do not repeat the completed audit.

The next release integrates on `develop`. Use small PRs targeting develop and separate worktrees outside this repository. The coordinator may delegate bounded work to subagents, review and merge passing PRs, and retire historical branches after verified preservation. Do not publish a release or launch paid compute as part of the foundation milestone.

Recipe configuration must remain flexible: ordinary Python components, explicit I/O, configurable objectives and regularizers. Warn on unqualified configurations and fail on actual incompatibilities. Never hide unsupported behavior behind successful no-ops or blanket test skips.

Older-checkpoint compatibility is not a resurrection requirement. Breaking checkpoint formats or source identities is acceptable; do not add migrations, compatibility shims or maintenance of old runtimes solely to support earlier implementations. Keep complete, validated save/resume and recovery from earlier snapshots within runs supported by the current implementation.

Update the status ledger at milestone boundaries and before a handoff/compaction. Record commands/results, PRs, blockers and the next concrete action. Preserve historical attribution and archive evidence before deleting legacy code or branches.
