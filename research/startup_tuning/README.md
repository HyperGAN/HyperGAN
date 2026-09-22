# Learning to learn quickly

This benchmark evaluates **startup tuning solutions**, separately from normal
HyperGAN training configurations. Establish controls before adding algorithms.
Production `--tune` is still experimental; this harness does not promote or
install a solution into training.

Latest: [2026-09-22 baseline results](results/2026-09-22-baseline/README.md).
Both original and half-rate controls failed the 32-update screen.

A [64px counterpart](testbeds/transgan64/README.md) uses the same rates and
schedule at lower resolution. Its screen is
[2026-09-22 64px baseline](results/2026-09-22-64px-baseline/README.md).
`configs/transgan-64-screen.json` is a separate manifest; keep its output root
separate. The 64px feature protocol
`online_dinov3_block11_spatial_mean_candidate_only_poly3_64px_4x4` is distinct
from the 128px protocol, so a 64px row is not a matched row against the 128px
leaderboard.

## Layout

- `configs/transgan-128-screen.json`: shared training-config reference, fixed
  evaluation horizon, observation steps, and an explicit finite list of solutions.
- `configs/transgan-64-screen.json`: the same solution files and horizon, pointed
  at the 64px training config. Keep its output root separate. Its feature
  protocol does not match the 128px leaderboard.
- `configs/solutions/*.json`: one versioned tuning solution per file. These are
  not training configs: they specify an algorithm and its options/evidence.
- `algorithms/`: local Python algorithms exposing `propose(context) -> dict`.
- `benchmark.py`: executes one solution, or the manifest's finite list, then
  regenerates a descriptive leaderboard.
- `proposals.py`: validates and applies owned initialization scales, G/D rates,
  and real per-layer optimizer rates, with full rollback.
- `leaderboard.py`: turns completed raw reports into JSON and Markdown tables.
- `PROTOCOL.md`: metric definitions, comparison requirements, and limitations.
- `results/`: tracked leaderboard snapshots and their compact raw reports.

Existing measurement implementations remain in `reports/joint_rate_probe.py`,
`reports/function_space_probe.py`, and `reports/frozen_feature_probe.py`. They
use the native training schedule, including lazy discriminator penalties.

## Run the baseline

From this worktree, with a training-capable Python environment:

```bash
CUDA_VISIBLE_DEVICES=GPU-548116b7-9dbe-de58-b3d9-a6e27b0f74ce \
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/martyn/dev/hypergan/training-runs/transgan-128-env/bin/python \
research/startup_tuning/benchmark.py \
research/startup_tuning/configs/transgan-128-screen.json \
--case all --device cuda:0 \
--output-root /mnt/ml7tb/hypergan-signal-research/startup-baseline-v1
```

Recheck GPU availability before execution. An existing case directory is never
overwritten. To add a candidate, add its solution path to a manifest and use
`--case its-id` in a fresh destination. The seed comes from the training config;
there is no seed sweep. The checked-in manifest references the local testbed;
on another machine, change that path and verify source/data/provider identities.

Each case writes `request.json`, a copy of `solution.json`, and an atomically
updated `report.json`. The request records manifest, solution, algorithm and
probe-evidence hashes; the report records source revision, actual resolved
parameters/rates, bank identity, metrics, cost, and restoration audits. These
JSON files contain no trained checkpoint. The output root holds
`leaderboard.json` and `leaderboard.md`.
The run also saves the original training TOML and resolved training JSON (with
network sources), plus the algorithm source for custom algorithms. Dataset and
pretrained artifacts remain external dependencies identified by the training
configuration and report hashes; they are not duplicated into benchmark results.

## Add a tuning solution

For an explicit hypothesis, copy a solution file and use `algorithm: "fixed"`:

```json
{
  "schema_version": 1,
  "id": "my-hypothesis",
  "algorithm": "fixed",
  "options": {
    "g_lr": 0.0001,
    "d_lr": 0.0001,
    "layer_lr_multipliers": [
      {"pattern": "graph.models.generator.network.nodes.n_stage8_block0_ffn.down.weight", "multiplier": 0.5}
    ]
  },
  "description": "Explicit hypothesis, not a validated recommendation"
}
```

`init_scales` uses the same pattern/multiplier format. A pattern must resolve to
owned trainable G/D parameters; zero matches, overlapping rules, buffers,
pretrained tensors, aliases, and prior parameters are rejected. Initialization
changes also synchronize their owned EMA counterparts. Layer rates use actual
optimizer parameter groups, not gradient multiplication that Adam can cancel.
Normal training config, prior rates, pretrained weights/buffers, and training
schedule remain protected. Per-layer groups exist only in disposable research
runs; this is not a persistent checkpoint format.

For a formula-based algorithm, point `algorithm` to a `.py` file relative to
the solution file. It exposes `propose(context)` where context contains a copy
of the resolved normal config, the solution options, and the JSON reports listed
in the solution's `evidence` paths. Return a proposal with `schema_version: 1`
and the same allowed fields shown above. It is called once, with no retries.
The algorithm receives no live trainer or benchmark monitor bank. Local plugin
code is trusted research code, not a security sandbox. Probe collection is a
separate, explicitly budgeted experiment; record its cost in the solution's
evidence. Cheap proposal computation alone is not the whole tuning cost.

## Agent workflow

1. Read `PROTOCOL.md` and the latest results. Check branch, clean state, GPU and
   data/provider availability. Keep PR #382 unmerged until the user approves it.
2. Start with controls on a single fixed seed. Never run the same experiment on
   a different seed. Do not extend a plainly failed control just to fill a table.
3. Collect bounded probes, state a formula and its assumptions, and sanity-test
   its predictions using unused probe points before evaluating a proposal.
4. Save the solution config and evidence paths/hashes. Test validation/rollback
   if adding a new intervention. Commit and push code before GPU measurement.
5. Run the same evaluator. Keep saturation, diversity and learning-proxy
   progress separate; neither small updates nor low saturation proves learning.
6. Update results with exact raw reports. Record failures and abstentions.
   Compare only matching protocol/data/initial-state identities and horizons.
7. Give a promising solution a separately declared longer tier (up to 512
   updates currently). A 32-update screen does not establish later stability.

Current measurement is one small **monitor bank**, possibly overlapping
training. DINO also participates in the critic. Its signed polynomial MMD² is
a noisy progress proxy, not independent sample-quality validation. Reproducible
inputs/provenance do not promise bitwise deterministic CUDA execution.
Comparison checks cover recorded conditions and the monitor-bank contents,
not every training-data pixel. For clean recorded commits available in Git,
the leaderboard derives a hash of evaluator/training source blobs: adding only
an algorithm or solution config does not invalidate an unchanged evaluator.
Changing evaluator code does require a new matched baseline. Full revision and
dependency provenance remain recorded; unavailable or dirty revisions use
strict source matching.
