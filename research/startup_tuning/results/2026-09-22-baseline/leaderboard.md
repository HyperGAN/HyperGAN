# Startup measurements

Designated baseline first, then input order; no ranking or aggregate score.

| Run | G / D / prior LR | Updates | Saturation initial → final | Output diversity initial → final (retention) | DINO MMD initial → final (Δ) | Feature spread retention | Sustained proxy step | Rollout seconds | Proposal seconds | Compared with baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline: source | 0.0002 / 0.0002 / 0.002 | 32/32 | 1.01% → 99.47% | 0.55186 → 0.0080666 (0.014617) | 0.33885 → 0.48586 (+0.14701) | 0.014677 | not observed | 111.83 | 1.0371e-05 | matched |
| comparison: half-rates | 0.0001 / 0.0001 / 0.002 | 32/32 | 1.01% → 98.96% | 0.55186 → 0.035712 (0.064713) | 0.33885 → 0.36123 (+0.022378) | 0.42267 | not observed | 96.642 | 1.006e-05 | matched |

Negative MMD Δ means a lower measured proxy. Retention is final / initial; zero or missing denominators remain undefined. Proxy steps are observations, not a claim of useful learning.

- **source**: seed `25002`, config SHA256 `2ff0cd0119a25b8cf5059a0ca12db06abe0db6dda88879545381b2b3d6cab137`, group `49e28ba15e906be0`, audit passed.
  Raw report: `/home/martyn/dev/hypergan/generator-signal-diagnostic/research/startup_tuning/results/2026-09-22-baseline/source/report.json`; SHA256 `bcc62920d95f782a5d392dd2a15f2fe7e46f15ced9b5f9bf281d5b84bfa74d20`.
  Budget: `{"completed_native_training_updates": 32, "feature_critic_forwards": 10, "feature_generator_forwards": 5, "feature_pretrained_forwards": 10, "feature_pretrained_images_including_gray_context": 1280, "frozen_feature_observations": 5, "rollout_observation_generator_forwards": 10}`.
  Evaluator tree SHA256 `4b81abdc98bb4c25cd157a130f9d44fd9dfd0ed4fcfa24f56a9ac463e7a6ec6a`, derived from recorded Git commit `74475eded4f7d82f5cdd53d1402cbbb68f23b844`.
  Recorded timing components (seconds): `{"diagnostic_seconds": 34.13563164201332, "observer_seconds_inside_update": 0.0, "preparation_seconds": 0.5546489169937558, "training_update_seconds": 62.44899010202789}`.
  Algorithm: `source`; proposal computation: 1.0371e-05 seconds.
  Solution: [source.json](</home/martyn/dev/hypergan/generator-signal-diagnostic/research/startup_tuning/configs/solutions/source.json>); SHA256 `0f845ad59d71fa5aee1555de8dd88cb6e56c457b8e83810ac5c167ccd4029afe`.
- **half-rates**: seed `25002`, config SHA256 `2ff0cd0119a25b8cf5059a0ca12db06abe0db6dda88879545381b2b3d6cab137`, group `49e28ba15e906be0`, audit passed.
  Raw report: `/home/martyn/dev/hypergan/generator-signal-diagnostic/research/startup_tuning/results/2026-09-22-baseline/half-rates/report.json`; SHA256 `2abbdcec5451f40644fe307c28e1b7bcdb1766cd46e23a563c99212283d779ac`.
  Budget: `{"completed_native_training_updates": 32, "feature_critic_forwards": 10, "feature_generator_forwards": 5, "feature_pretrained_forwards": 10, "feature_pretrained_images_including_gray_context": 1280, "frozen_feature_observations": 5, "rollout_observation_generator_forwards": 10}`.
  Evaluator tree SHA256 `4b81abdc98bb4c25cd157a130f9d44fd9dfd0ed4fcfa24f56a9ac463e7a6ec6a`, derived from recorded Git commit `74475eded4f7d82f5cdd53d1402cbbb68f23b844`.
  Recorded timing components (seconds): `{"diagnostic_seconds": 29.78960994099907, "observer_seconds_inside_update": 0.0, "preparation_seconds": 0.250871587995789, "training_update_seconds": 55.525586758958525}`.
  Algorithm: `fixed`; proposal computation: 1.006e-05 seconds.
  Solution: [half-rates.json](</home/martyn/dev/hypergan/generator-signal-diagnostic/research/startup_tuning/configs/solutions/half-rates.json>); SHA256 `bd18990348aec32f57bfb70648578a1333649e802aacd0ee02e6b99521341795`.

- Frozen-DINO polynomial MMD is a noisy small-bank proxy, not Inception KID, statistical significance, or a quality/convergence certificate. Negative unbiased estimates are valid.
- Sustained means strictly below the initial MMD at every remaining observed checkpoint, with at least two such checkpoints. It says nothing about unobserved steps or behavior beyond the measured horizon.
- Image and feature diversity retention describe contraction or expansion; neither proves useful learning. Saturation is the output fraction with absolute value above 0.99.
- The online generator and evolving learned prior are measured together. Fixed-latent statistics are included separately to isolate generator motion.
- Total wall time includes setup, measurements and optional diagnostics. It is not training-only throughput or time to learn.
- Rollout elapsed time includes applying the proposal but excludes computing it. Proposal computation is recorded separately; producing prior evidence reports is an additional, generally unaccounted cost. Neither number is full research cost.
- Matching seed alone does not establish matching evaluation data. Missing protocol or evaluation-bank identity makes comparison unverified; different horizons remain separate.
- Clean recorded Git commits may differ when their training/evaluator trees match. That digest is derived from the recorded commit, never current working files; dependency provenance remains part of comparison identity. Dirty or unavailable commits retain strict source identity.
