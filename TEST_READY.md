# TEST_READY: ImaJev-4B Constraint GAN E2E Test Suite & Quantitative Evaluation Harness

## 1. Readiness Summary

The complete opaque-box E2E test suite and turnkey quantitative evaluation harness for the **ImaJev-4B Constraint GAN** have been fully implemented, verified, and published.

All deliverables requested in the E2E Testing Track dispatch are complete:
1. **Test Infrastructure Specification**: `/nvme/dev/hypergan/HyperGAN/TEST_INFRA.md`
   - Defines opaque-box, requirement-driven test philosophy.
   - Comprehensive inventory mapping across all 21 features in `PROJECT.md`.
   - Methodologies: Category-Partition, Boundary Value Analysis, Pairwise Combinations, Real-World Application Scenarios.
   - Minimum coverage thresholds across Tiers 1–4.
2. **Turnkey Quantitative Acceptance Verification Script**: `/nvme/dev/hypergan/HyperGAN/scripts/verify_imajev_acceptance.py`
   - Evaluates $\ge 64$ samples across all 4 acceptance criteria dimensions.
   - Full CLI argument support: `--checkpoint`, `--recipe`, `--samples`, `--device`, `--output-dir`, `--json`, `--synthetic`, `--scenario`.
   - Structured JSON reporting and formatted console summaries.
3. **Automated E2E Playbook Suite**: `/nvme/dev/hypergan/HyperGAN/tests/playbooks/test_playbook_imajev_e2e.py`
   - Complete test suite testing all metric functions, boundary inputs, failure scenarios, and CLI workflows across Tiers 1–4.

---

## 2. Acceptance Criteria Thresholds & Invariants

| Dimension | Metric | Required Threshold | Verification Function |
|-----------|--------|--------------------|-----------------------|
| **In-Distribution Joint Satisfaction** | Mean target mode probability across all 6 queries | $\ge 0.70$ ($70\%$) | `compute_in_distribution_satisfaction` |
| | Mean noise/static option probability | $\le 0.05$ ($5\%$) | `compute_in_distribution_satisfaction` |
| | Mean unknown option probability | $\le 0.08$ ($8\%$) | `compute_in_distribution_satisfaction` |
| **Held-Out Audit Robustness** | Top-1 apple accuracy on $\ge 3$ held-out queries under shifts, crops, and 4 cyclic rotations | $\ge 75\%$ | `compute_held_out_audit_accuracy` |
| | Combined artifact/noise/unknown option probability | $\le 12\%$ | `compute_held_out_audit_accuracy` |
| **Multi-Mode Balance** | Empirical marginal Shannon entropy ratio $H(p) / \ln(M_{\text{modes}})$ | $\ge 0.85$ ($85\%$) | `compute_mode_marginal_entropy` |
| **Perceptual Diversity** | Mean pairwise cosine distance of patch-pooled vision tower features | $\ge 0.12$ | `compute_pairwise_cosine_diversity` |
| **Optical Coherence** | ViT patch-seam gradient ratio $\mathcal{R}_{\text{seam}}$ | $\le 1.15$ | `compute_patch_seam_gradient_ratio` |
| **Spatial Invariance** | Sub-patch $(4, 4)$ pixel shift sensitivity $\Delta p_{\text{target}}$ | $< 0.12$ | `compute_shift_sensitivity` |

---

## 3. Playbook Test Suite Inventory (`tests/playbooks/test_playbook_imajev_e2e.py`)

- **Tier 1: Core Mathematical Metrics & Contracts**
  - `test_tier1_in_distribution_satisfaction_math`: Verifies simultaneous 6-query satisfaction mathematics.
  - `test_tier1_held_out_audit_accuracy_math`: Verifies top-1 apple accuracy and artifact suppression on held-out queries.
  - `test_tier1_mode_marginal_entropy_uniform_and_collapsed`: Verifies Shannon entropy calculation on uniform vs collapsed modes.
  - `test_tier1_pairwise_cosine_diversity`: Verifies cosine distance on orthogonal and identical features.
  - `test_tier1_patch_seam_gradient_ratio_smooth_vs_spiked`: Verifies seam ratio on smooth vs boundary-spiked images.
  - `test_tier1_shift_sensitivity_invariance`: Verifies shift sensitivity detection on small vs large probability shifts.
- **Tier 2: Boundary Value Analysis & Degenerate Tensors**
  - `test_tier2_boundary_sample_counts`: Asserts rejection of $N=63$ samples and acceptance of $N=64$.
  - `test_tier2_probability_threshold_boundaries`: Tests exact threshold boundaries ($0.700$ vs $0.699$, $0.050$ vs $0.051$, $0.080$ vs $0.081$).
  - `test_tier2_degenerate_and_invalid_inputs`: Asserts clean validation errors on zero-gradient, non-4D, non-RGB, and single-mode inputs.
- **Tier 3: Combinations & Anti-Goodharting**
  - `test_tier3_spatial_and_cyclic_transformations`: Verifies $(4, 4)$ translation, 90% crop, and 4 cyclic option permutations.
  - `test_tier3_failure_scenario_classification`: Verifies distinct failure classification for mode collapse, patch static, and noise.
- **Tier 4: Turnkey CLI Execution & Mock Checkpoint Integration**
  - `test_tier4_cli_standalone_synthetic_pass`: Validates CLI execution with exit code 0 and valid JSON schema.
  - `test_tier4_cli_synthetic_failure_exit_code`: Validates CLI execution with exit code 1 on failed acceptance criteria.
  - `test_tier4_cli_boundary_sample_count_rejection`: Validates CLI execution with exit code 2 on sample count $< 64$.
  - `test_tier4_cli_output_dir_report_generation`: Validates writing `imajev_acceptance_report.json` to specified output directory.
  - `test_tier4_mock_checkpoint_loading_and_evaluation`: Validates loading and evaluating a mock generator checkpoint file.

---

## 4. How to Run

### Run Full E2E Playbook Test Suite
```bash
python -m pytest tests/playbooks/test_playbook_imajev_e2e.py -v
```

### Run Turnkey Verification Script (Synthetic Baseline)
```bash
python scripts/verify_imajev_acceptance.py --samples 64 --synthetic --json
```

### Run Turnkey Verification Script (Trained Checkpoint)
```bash
python scripts/verify_imajev_acceptance.py \
    --checkpoint <path_to_checkpoint.pt> \
    --recipe examples/imajev-apples-joint-bispf.toml \
    --samples 64 \
    --device auto \
    --output-dir reports/imajev_acceptance \
    --json
```
