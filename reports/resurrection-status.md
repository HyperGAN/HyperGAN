# Resurrection execution ledger

Authoritative design: [resurrection plan](resurrecting-hypergan-plan-2026-09-18.md). Updated 2026-09-18 (America/Denver).

## Accepted decisions

- Integrate the next release on existing `develop`; coordinator reviews and merges passing PRs. Feature work uses external worktrees and subagents.
- First checkpoint: clean package/CI, flexible recipe configuration, bounded ParticleGAN CPU reference, inference reload, legacy retirement and verified archives.
- Configuration supports custom generator/discriminator/encoder/auxiliary components, I/O bindings, losses and regularizers. Warn on unqualified combinations; reject actual incompatibilities. Default b-cap and VICReg follow the pinned upstream reference.
- Colorization and super-resolution influence conditional I/O now; real image recipes, complete resume and distributed qualification follow.
- No paid compute or release publishing this session. Reserve $30 Modal credit for later cluster validation; additional spending needs a concrete agreed allocation.
- Next package line: 2.0.0a1 (development only). ParticleGAN 0.5.0 initial dependency; root license declaration remains a release prerequisite.

## Tasks

| ID | Task / owner | Dependencies | State and evidence |
| --- | --- | --- | --- |
| F1 | Preserve history / branch agent | none | Active: external archives and restore verification |
| F2 | Bootstrap develop / coordinator | verified checks | PR #299: preserve master ancestry; replace stale required CircleCI check after Actions passes |
| F3 | Plan and continuity / coordinator | F2 | PR #298 will target develop; this ledger and AGENTS.md |
| F4 | Packaging, CLI, CI / package agent | F2 | `resurrection/package-foundation`, external `resurrection-worktrees/package` |
| F5 | Configuration and CPU reference / reference agent | F4 interface | `resurrection/recipe-reference`, external `resurrection-worktrees/reference` |
| F6 | Legacy code and branch retirement / coordinator | F1, working F4+F5 | Pending |
| F7 | Integrated acceptance and handoff / coordinator | F1–F6 | Pending |

## Acceptance evidence

Implementation gates are pending. The earlier audit's 143 upstream test passes are upstream evidence, not tests of the new HyperGAN code. Bootstrap integrity checks validate history/attribution only and do not establish runtime support.

## Resume here

Check live PRs #298/#299 and agent worktrees; complete bootstrap, then integrate packaging and reference changes in dependency order. Require fresh-install CLI tests, configuration/paired-I/O tests, numerical update parity, finite reference execution, fresh-process inference reload, and clean wheel/sdist contents before the foundation checkpoint is complete.

Follow-on checkpoint: image workflow, full atomic training checkpoint/resume, and two-process CPU objective/state parity, then actual two-GPU and two-node qualification.
