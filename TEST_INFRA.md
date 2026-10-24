# Test Infrastructure: ImaJev-4B Constraint GAN E2E Testing Suite

## 1. Test Philosophy: Opaque-Box & Requirement-Driven

The testing infrastructure for the **ImaJev-4B Constraint GAN** operates under strict **opaque-box, requirement-driven principles**:
1. **Opaque-Box Evaluation**: The test suite evaluates observable inputs, outputs, metric distributions, and mathematical invariants without depending on internal private state, weight layout details, or arbitrary hidden variables.
2. **Authoritative Specification Anchoring**: Every expected threshold, boundary limit, and metric formula is derived directly from the authoritative specifications in `PROJECT.md` and `ORIGINAL_REQUEST.md` (2026-10-23 release).
3. **Progressive Testability & Isolation**: Tests are self-contained, idempotent, and partitioned into clean tiers. They can run against both simulated/synthetic generator outputs (for deterministic, fast CPU CI execution) and live trained checkpoints on GPU hardware.
4. **Anti-Goodharting & Adversarial Defense**: Evaluation does not merely measure training loss convergence; it actively tests against shortcut exploitation (e.g. prompt overfitting, high-frequency checkerboard ViT grid attacks, and batch collapse) via held-out audit queries, spatial sub-patch shifts, random crops, and cyclic option rotations.

---

## 2. Feature Inventory Mapping (All 21 Features in PROJECT.md)

| # | Feature Key | Description | Milestone | Verification Tier & Strategy |
|---|-------------|-------------|-----------|------------------------------|
| 1 | `web_autostart_watermark_fix` | Restore `is_active` check precedence in `Projector._should_project` | M1 | Tier 1: Unit & integration test verifying projector idle status precedence |
| 2 | `model_file_detached_monitor_fix` | Allow detached monitor objectives (`detach = ["logits"]`) in `_validate_loss_graph` | M1 | Tier 1: Recipe graph validation rejecting false positive errors on detached monitor nodes |
| 3 | `qwen_32_layer_indexing_fix` | Correct layer count indexing to 32 layers (`[-32, 31]`) in provider and config | M1 | Tier 2: Boundary test validating bounds `[-32, 31]` and rejecting out-of-bounds layer indices |
| 4 | `git_whitespace_clean` | Fix trailing blank line at EOF in `tests/reference/test_constraint_data.py` | M1 | Tier 1: `git diff --check` and lint hygiene checks |
| 5 | `vision_once_caching` | Cache ViT features across all $Q$ queries per step, avoiding redundant vision passes | M2 | Tier 1 & 3: Profile forward execution verifying exactly 1 vision tower pass per batch |
| 6 | `joint_multi_query_scoring` | Simultaneous 6-query batched scoring (`joint_log_odds`), replacing queue | M2 | Tier 1: Verify output shape `[B, Q * K_max]` and differentiable backprop |
| 7 | `calibrated_soft_capped_log_odds` | Apply $T = 1.30515697$ and $\tau \tanh(u / \tau)$ ($\tau = 3.0$) soft capping | M2 | Tier 1 & 2: Mathematical verification of bounded outputs in $[-\tau, \tau]$ |
| 8 | `deprecate_broken_nullspace_and_queue` | Remove `_IMAJEV_ACTIVE_QUERY_QUEUE` and broken `h_perp` nullspace code | M2 | Tier 1: Verify no global state leaks and removal of fragile nullspace projections |
| 9 | `sobolev_gradient_preconditioning` | Riemannian Sobolev $H^1$ gradient filtering cutting high-frequency static | M3 | Tier 1 & 3: Spectral energy falloff verification on backpropagated image gradients |
| 10 | `optical_wavefront_prior` | Objective penalizing ViT 8px/16px patch-seam ratio, biharmonic curvature, $1/f$ | M3 | Tier 1 & 2: Numerical validation of $\mathcal{R}_{\text{seam}} \le 1.15$ penalty and curvature |
| 11 | `on_manifold_particle_diversity` | Marginal entropy loss + normalized vision feature cosine repulsion | M3 | Tier 1 & 3: Measure marginal mode entropy and pairwise cosine distance repulsion |
| 12 | `multiscale_pyramid_generator` | Upgraded HNDL generator with smooth multi-scale bicubic synthesis | M3 | Tier 1: Forward synthesis test across resolutions (64x64, 128x128) |
| 13 | `upgraded_apple_recipe` | Complete TOML recipe integrating joint scoring, optical priors, and particle prior | M4 | Tier 1 & 3: Recipe syntax and runtime graph validation |
| 14 | `gpu_training_convergence` | Train ImaJev-4B constraint GAN on RTX A6000 GPU to convergence | M4 | Tier 4: Hardware convergence qualification |
| 15 | `standalone_verification_harness` | Turnkey `scripts/verify_imajev_acceptance.py` script evaluating all acceptance criteria | M5 / E2E | Tier 1-4: CLI execution, argument parsing, JSON schema, and exit codes |
| 16 | `in_distribution_multi_query_eval` | Verify $\ge 0.70$ target mode prob, $\le 0.05$ noise, $\le 0.08$ unknown across 6 queries | M5 / E2E | Tier 1 & 4: Quantitative evaluation on $\ge 64$ samples |
| 17 | `held_out_audit_query_eval` | Verify 3 held-out queries: $\ge 75\%$ apple accuracy, $\le 12\%$ noise/unknown under shift/crop/rotation | M5 / E2E | Tier 1, 3, 4: Transformation robustness and anti-Goodharting evaluation |
| 18 | `mode_entropy_and_diversity_eval` | Verify entropy $\ge 0.85 \ln(M)$ and pairwise cosine feature distance $\ge 0.12$ | M5 / E2E | Tier 1 & 2: Mathematical property and diversity metrics across batch |
| 19 | `patch_seam_and_shift_eval` | Verify patch-seam ratio $\le 1.15$ and $(4,4)$ pixel shift sensitivity $< 0.12$ | M5 / E2E | Tier 1 & 2: ViT boundary gradient ratio and spatial translation sensitivity |
| 20 | `e2e_opaque_box_test_suite` | Complete test suite across Tiers 1-4 with `TEST_READY.md` | E2E | Tier 1-4: Full regression test execution in `test_playbook_imajev_e2e.py` |
| 21 | `adversarial_coverage_hardening` | Tier 5 white-box challenger analysis and adversarial edge-case tests | M5 | Tier 5: Adversarial edge cases, degenerate tensors, boundary inputs |

---

## 3. Testing Methodology

The test suite systematically applies four established testing methodologies:

### A. Category-Partition Testing
Inputs, configurations, and runtime environments are partitioned into equivalence classes:
1. **Model Source Categories**: Real trained checkpoint (`.pt` bundle), mock checkpoint dictionary, synthetic generator function, pre-generated image directory.
2. **Device Categories**: `cpu`, `cuda:0`, `auto` fallback.
3. **Query Categories**:
   - In-distribution training queries ($Q=6$: `subject`, `color`, `bites`, `sitting_on`, `style`, `stem_leaf`).
   - Held-out audit queries ($Q \ge 3$: 3D volume/geometry, object completeness/context, organic skin texture).
4. **Transform Categories**:
   - Identity (untransformed).
   - Sub-patch spatial translation (e.g., $\pm 4$px in $x$ and $y$).
   - Random/center spatial crops ($90\%$ scale resized to input resolution).
   - Cyclic option order permutations (4 cyclic rotations: $r \in \{0, 1, 2, 3\}$).
5. **Image Distribution Categories**:
   - High-fidelity synthetic manifold samples.
   - High-frequency adversarial checkerboard / ViT patch-seam static.
   - Single-color / uniform mode collapsed particles.
   - Gaussian noise / out-of-distribution inputs.

### B. Boundary Value Analysis (BVA)
Boundary values are tested at and around critical specification thresholds:
1. **Sample Count $N$**:
   - $N = 63$ (below minimum threshold: triggers validation error).
   - $N = 64$ (exact threshold boundary: valid).
   - $N = 128$ (large batch: valid).
2. **In-Distribution Probabilities**:
   - Target mode probability $p_{\text{target}}$: boundary at $0.70$ ($0.699$ fail, $0.701$ pass).
   - Noise probability $p_{\text{noise}}$: boundary at $0.05$ ($0.051$ fail, $0.049$ pass).
   - Unknown probability $p_{\text{unknown}}$: boundary at $0.08$ ($0.081$ fail, $0.079$ pass).
3. **Held-Out Audit Query Robustness**:
   - Apple option top-1 accuracy: boundary at $0.75$ ($74.9\%$ fail, $75.1\%$ pass).
   - Artifact / noise / unknown combined probability: boundary at $0.12$ ($0.121$ fail, $0.119$ pass).
4. **Multi-Mode Entropy & Feature Diversity**:
   - Mode entropy ratio: boundary at $0.85 \ln(M_{\text{modes}})$.
   - Normalized feature pairwise cosine distance: boundary at $0.12$ ($0.119$ fail, $0.121$ pass).
5. **Optical & Spatial Coherence**:
   - ViT patch-seam gradient ratio $\mathcal{R}_{\text{seam}}$: boundary at $1.15$ ($1.151$ fail, $1.149$ pass).
   - Sub-patch $(4, 4)$ shift sensitivity $\Delta p$: boundary at $0.12$ ($0.121$ fail, $0.119$ pass).
6. **Input Tensor Ranges & Shapes**:
   - Pixel range boundaries: $[-1.0, 1.0]$, $[0.0, 1.0]$, out-of-range clipping/handling.
   - Spatial resolutions: $64 \times 64$, $128 \times 128$, non-multiples of 32 rejection.

### C. Pairwise Combinations
Systematic combinatorial coverage across evaluation options:
- [Device (`cpu`, `cuda`)] $\times$ [Checkpoint type (Mock, File, Function)] $\times$ [Format (`--json`, standard)] $\times$ [Sample size ($64$, $128$)] $\times$ [Batch size ($16$, $32$, $64$)].

### D. Real-World Application Scenarios
1. **Standalone CLI Workflow**: CLI execution of `python scripts/verify_imajev_acceptance.py --checkpoint ... --json` reproducing human auditor evaluation.
2. **Automated CI/Playbook Workflow**: Programmatic invocation via `tests/playbooks/test_playbook_imajev_e2e.py` without requiring manual intervention.
3. **Failure Classification & Diagnosis**: Structured error messages when generated samples fail specific criteria (e.g. diagnosing whether failure was mode collapse, patch artifacts, or semantic misalignment).

---

## 4. The 4-Tier Test Pyramid

```
                      / \
                     /   \
                    / T4  \  Tier 4: Real Checkpoint & GPU Qualification
                   /-------\ (>=64 samples on RTX A6000, multi-query VLM passes)
                  /   T3    \ Tier 3: Combinations & Anti-Goodharting
                 /-----------\ (spatial shift, crop, cyclic option rotations)
                /     T2      \ Tier 2: Boundary Value Analysis & Degenerate Tensors
               /---------------\ (N=63 vs 64, seam ratio 1.15, collapse, noise)
              /       T1        \ Tier 1: Core Mathematical Metrics & Contract API
             /-------------------\ (6-query parsing, entropy, cosine dist, CLI schema)
```

### Minimum Coverage Thresholds Across Tiers 1–4

| Tier | Focus Area | Required Pass Rate | Target Invariants |
|------|------------|--------------------|-------------------|
| **Tier 1** | Core Functions & API Contracts | 100% | In-distribution satisfaction, entropy formula, pairwise distance, seam ratio, shift sensitivity return exact mathematical results on synthetic tensors. |
| **Tier 2** | Boundary Values & Edge Inputs | 100% | Rejection of $N < 64$; correct behavior on degenerate constant images, pure noise, and boundary probabilities ($0.70, 0.05, 0.08, 0.12, 1.15$). |
| **Tier 3** | Combinations & Perturbations | 100% | Transformation invariance: $(4, 4)$ shift, crops, 4 cyclic option rotations test anti-shortcut robustness. |
| **Tier 4** | Turnkey Script & System End-to-End | 100% | `verify_imajev_acceptance.py` CLI runs cleanly with valid exit codes, produces complete JSON summaries, and evaluates full candidate batches. |

---

## 5. Test Execution Environment & Commands

### Working Directory
Target script: `/nvme/dev/hypergan/HyperGAN/scripts/verify_imajev_acceptance.py`
Target playbook: `/nvme/dev/hypergan/HyperGAN/tests/playbooks/test_playbook_imajev_e2e.py`

### Python Execution Commands
- **Run the E2E Playbook Test Suite**:
  ```bash
  pytest tests/playbooks/test_playbook_imajev_e2e.py -v
  ```

- **Run Specific Tier Tests**:
  ```bash
  pytest tests/playbooks/test_playbook_imajev_e2e.py -k "tier1" -v
  pytest tests/playbooks/test_playbook_imajev_e2e.py -k "tier2" -v
  pytest tests/playbooks/test_playbook_imajev_e2e.py -k "tier3" -v
  pytest tests/playbooks/test_playbook_imajev_e2e.py -k "tier4" -v
  ```

- **Run the Turnkey Verification Script (Standalone CLI)**:
  ```bash
  python scripts/verify_imajev_acceptance.py --samples 64 --device auto --json
  ```

---

## 6. Expected Output Derivation & Authoritative Sources

| Test / Feature | Authoritative Source | Derivation Method |
|---|---|---|
| In-Distribution Joint Satisfaction | `PROJECT.md` § Feature 16 & AC 1 | 6 queries (`subject`, `color`, `bites`, `sitting_on`, `style`, `stem_leaf`), mean target prob $\ge 0.70$, noise prob $\le 0.05$, unknown prob $\le 0.08$. |
| Held-Out Audit Query Accuracy | `PROJECT.md` § Feature 17 & AC 1 | $\ge 3$ held-out queries under $\pm 4$px shifts, crops, and 4 cyclic rotations: top-1 apple accuracy $\ge 75\%$, noise/unknown $\le 12\%$. |
| Multi-Mode Entropy | `PROJECT.md` § Feature 18 & AC 1 | Shannon entropy $H = -\sum p_m \ln p_m \ge 0.85 \ln(M_{\text{modes}})$. |
| Perceptual Feature Diversity | `PROJECT.md` § Feature 18 & AC 1 | Mean pairwise cosine distance of normalized ViT features $\ge 0.12$. |
| ViT Patch-Seam Gradient Ratio | `PROJECT.md` § Feature 19 & AC 1 | Ratio of mean gradient along $8\text{px}/16\text{px}$ seams to interior gradient $\le 1.15$. |
| Spatial Shift Sensitivity | `PROJECT.md` § Feature 19 & AC 1 | Mean probability variation under $(4, 4)$ pixel shift $< 0.12$. |

---

## 7. Isolation, Teardown & Hygiene Guarantees

1. **Ephemeral Output Directories**: All CLI tests outputting logs or artifacts write to pytest `tmp_path`.
2. **Stateless Metric Evaluation**: All metric functions are pure functions of inputs and do not mutate global variables.
3. **Deterministic Random Generators**: All synthetic batches utilize seeded `torch.Generator` or `numpy.random.RandomState`.
4. **Memory Hygiene**: GPU test runs empty CUDA cache before and after batch evaluations.
