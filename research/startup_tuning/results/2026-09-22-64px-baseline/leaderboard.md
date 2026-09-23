# Startup measurements

Designated baseline first, then input order; no ranking or aggregate score.

| Run | G / D / prior LR | Updates | Saturation initial → final | Output diversity initial → final (retention) | DINO MMD initial → final (Δ) | Feature spread retention | Sustained proxy step | Rollout seconds | Proposal seconds | Compared with baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline: source | 0.0002 / 0.0002 / 0.002 | 32/32 | 2.62% → 100.00% | 0.58408 → 1.3797e-07 (2.3623e-07) | 0.2873 → 0.48519 (+0.19789) | 0.0014589 | not observed | 76.479 | 8.44e-06 | matched |
| comparison: half-rates | 0.0001 / 0.0001 / 0.002 | 32/32 | 2.62% → 71.26% | 0.58408 → 0.22975 (0.39335) | 0.2873 → 0.40064 (+0.11335) | 0.5402 | not observed | 68.178 | 9.46e-06 | matched |

Negative MMD Δ means a lower measured proxy. Retention is final / initial; zero or missing denominators remain undefined. Proxy steps are observations, not a claim of useful learning.

- **source**: seed `25002`, config SHA256 `9b70e9f9a449f5744d6517124557dfb26a230b61ede8621a94a062d50e115f61`, group `e5bdbbde122ada37`, audit passed.
  Raw report: `/home/martyn/dev/hypergan/generator-signal-diagnostic/research/startup_tuning/results/2026-09-22-64px-baseline/source/report.json`; SHA256 `0797a9f463e758f2e37141928ae919c5704fe276494eb79815ae0461dd9c42c1`.
  Budget: `{"completed_native_training_updates": 32, "feature_critic_forwards": 10, "feature_generator_forwards": 5, "feature_pretrained_forwards": 10, "feature_pretrained_images_including_gray_context": 1280, "frozen_feature_observations": 5, "rollout_observation_generator_forwards": 10}`.
  Recorded timing components (seconds): `{"diagnostic_seconds": 27.844600390977575, "observer_seconds_inside_update": 0.0, "preparation_seconds": 0.49759782299224753, "training_update_seconds": 34.927985533038736}`.
  Algorithm: `source`; proposal computation: 8.44e-06 seconds.
  Solution: [source.json](</home/martyn/dev/hypergan/generator-signal-diagnostic/research/startup_tuning/configs/solutions/source.json>); SHA256 `0f845ad59d71fa5aee1555de8dd88cb6e56c457b8e83810ac5c167ccd4029afe`.
- **half-rates**: seed `25002`, config SHA256 `9b70e9f9a449f5744d6517124557dfb26a230b61ede8621a94a062d50e115f61`, group `e5bdbbde122ada37`, audit passed.
  Raw report: `/home/martyn/dev/hypergan/generator-signal-diagnostic/research/startup_tuning/results/2026-09-22-64px-baseline/half-rates/report.json`; SHA256 `8de67fa0609d2a6c24fb9b11d99d01284dac2b7d92f6fb47cd0255a02c15a76c`.
  Budget: `{"completed_native_training_updates": 32, "feature_critic_forwards": 10, "feature_generator_forwards": 5, "feature_pretrained_forwards": 10, "feature_pretrained_images_including_gray_context": 1280, "frozen_feature_observations": 5, "rollout_observation_generator_forwards": 10}`.
  Recorded timing components (seconds): `{"diagnostic_seconds": 24.634897127005388, "observer_seconds_inside_update": 0.0, "preparation_seconds": 0.23583534600038547, "training_update_seconds": 32.89565286497236}`.
  Algorithm: `fixed`; proposal computation: 9.46e-06 seconds.
  Solution: [half-rates.json](</home/martyn/dev/hypergan/generator-signal-diagnostic/research/startup_tuning/configs/solutions/half-rates.json>); SHA256 `bd18990348aec32f57bfb70648578a1333649e802aacd0ee02e6b99521341795`.

- Frozen-DINO polynomial MMD is a noisy small-bank proxy, not Inception KID, statistical significance, or a quality/convergence certificate. Negative unbiased estimates are valid.
- Sustained means strictly below the initial MMD at every remaining observed checkpoint, with at least two such checkpoints. It says nothing about unobserved steps or behavior beyond the measured horizon.
- Image and feature diversity retention describe contraction or expansion; neither proves useful learning. Saturation is the output fraction with absolute value above 0.99.
- The online generator and evolving learned prior are measured together. Fixed-latent statistics are included separately to isolate generator motion.
- Total wall time includes setup, measurements and optional diagnostics. It is not training-only throughput or time to learn.
- Rollout elapsed time includes applying the proposal but excludes computing it. Proposal computation is recorded separately; producing prior evidence reports is an additional, generally unaccounted cost. Neither number is full research cost.
- Matching seed alone does not establish matching evaluation data. Missing protocol or evaluation-bank identity makes comparison unverified; different horizons remain separate.
- Clean recorded Git commits may differ when their training/evaluator trees match. That digest is derived from the recorded commit, never current working files; dependency provenance remains part of comparison identity. Dirty or unavailable commits retain strict source identity.
