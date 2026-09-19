# Contributing

The next HyperGAN release integrates on `develop`. Read [the execution ledger](reports/resurrection-status.md) and [AGENTS.md](AGENTS.md) before starting work. Create a short-lived branch from current develop and target your pull request at develop.

Keep PRs focused on a working user outcome or a bounded correctness improvement. Describe the behavior changed, relevant validation and remaining limits. Passing required CI and coordinator review are required before integration. Use separate worktrees outside the repository for parallel agent work.

Recipe configuration supports custom Python components. Document each component's inputs, outputs, parameters and numerical assumptions. Unknown configurations should receive an honest qualification warning; invalid inputs and unsupported capabilities must fail clearly. Do not silently ignore parameters, replace components, or imply a successful run demonstrates image quality or distributed correctness.

Research enters the qualified catalog only after reproducible evidence and product validation. See the research-admission criteria in the plan. Historical implementations remain accessible through archive tags; avoid reintroducing the legacy catalog or its dependencies wholesale.

Use a fresh environment and run the installation, configuration and reference tests specified by the current CI workflow. Lightweight commands must work without the training dependencies. CPU tests precede local GPU qualification; paid cluster checks are deliberately scheduled and are not automatic PR jobs.
