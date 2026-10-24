"""UX & E2E Playbook: ImaJev-4B Constraint GAN Turnkey Acceptance Verification.

Automated opaque-box E2E test suite covering Tiers 1-4:
- Tier 1: Core Mathematical Metrics & Contract APIs (In-distribution satisfaction,
  held-out audit accuracy, multi-mode entropy, feature diversity, patch-seam ratio, shift sensitivity).
- Tier 2: Boundary Value Analysis & Degenerate Tensors (N=63 vs N=64, boundary probabilities,
  zero gradients, invalid shapes/channels).
- Tier 3: Combinations & Anti-Goodharting (Spatial shifts, crops, cyclic option rotations,
  failure classification across collapse, patch_static, and noise scenarios).
- Tier 4: Turnkey CLI Execution & Mock Checkpoint Integration (CLI flags, JSON schema,
  subprocess exit codes, mock generator checkpoint loading).
"""

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

import scripts.verify_imajev_acceptance as verifier

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "verify_imajev_acceptance.py"


# ==============================================================================
# Tier 1: Core Mathematical Metrics & Interface Contracts
# ==============================================================================

def test_tier1_in_distribution_satisfaction_math():
    """Verify in-distribution satisfaction calculation across 6 semantic queries."""
    N = 64
    probs = {}
    for q in verifier.IN_DISTRIBUTION_QUERIES:
        aspect = q["aspect"]
        K = len(q["options"])
        p = torch.zeros(N, K)
        p[:, q["valid_target_modes"][0]] = 0.90
        p[:, q["noise_index"]] = 0.04
        p[:, q["unknown_index"]] = 0.06
        probs[aspect] = p

    res = verifier.compute_in_distribution_satisfaction(probs)
    assert res["passed"] is True
    assert res["mean_target_prob"] == pytest.approx(0.90, abs=1e-5)
    assert res["mean_noise_prob"] == pytest.approx(0.04, abs=1e-5)
    assert res["mean_unknown_prob"] == pytest.approx(0.06, abs=1e-5)
    assert len(res["per_query"]) == 6
    for aspect in ["subject", "color", "bites", "sitting_on", "style", "stem_leaf"]:
        assert aspect in res["per_query"]
        assert res["per_query"][aspect]["top1_target_accuracy"] == 1.0


def test_tier1_held_out_audit_accuracy_math():
    """Verify held-out audit accuracy and artifact suppression on 3 audit queries."""
    N = 64
    audit_probs = {}
    for q in verifier.HELD_OUT_AUDIT_QUERIES:
        aspect = q["aspect"]
        K = len(q["options"])
        p = torch.zeros(N, K)
        p[:, q["valid_apple_index"]] = 0.80  # Apple option top-1
        p[:, q["artifact_noise_indices"][0]] = 0.04
        p[:, q["unknown_index"]] = 0.05
        # Fill remaining probability into reject option
        reject_indices = [
            i for i in range(K)
            if i != q["valid_apple_index"]
            and i not in q["artifact_noise_indices"]
            and i != q["unknown_index"]
        ]
        if reject_indices:
            p[:, reject_indices[0]] = 0.11
        audit_probs[aspect] = p

    res = verifier.compute_held_out_audit_accuracy(audit_probs)
    assert res["passed"] is True
    assert res["apple_accuracy"] == pytest.approx(1.0, abs=1e-5)
    assert res["artifact_noise_unknown_prob"] == pytest.approx(0.09, abs=1e-5)
    assert res["passed_accuracy"] is True
    assert res["passed_artifact"] is True


def test_tier1_mode_marginal_entropy_uniform_and_collapsed():
    """Verify empirical Shannon entropy H = -sum p ln p on uniform and collapsed distributions."""
    N = 100
    M = 3
    # 1. Uniform 3-mode distribution: H should equal ln(3)
    uniform_p = torch.full((N, M), 1.0 / M)
    res_uniform = verifier.compute_mode_marginal_entropy(uniform_p, num_modes=M)
    assert res_uniform["passed"] is True
    assert res_uniform["empirical_entropy"] == pytest.approx(math.log(3), abs=1e-5)
    assert res_uniform["entropy_ratio"] == pytest.approx(1.0, abs=1e-5)

    # 2. Completely collapsed distribution: all mass in mode 0 -> H = 0
    collapsed_p = torch.zeros(N, M)
    collapsed_p[:, 0] = 1.0
    res_collapsed = verifier.compute_mode_marginal_entropy(collapsed_p, num_modes=M)
    assert res_collapsed["passed"] is False
    assert res_collapsed["empirical_entropy"] < 1e-6
    assert res_collapsed["entropy_ratio"] < 1e-6


def test_tier1_pairwise_cosine_diversity():
    """Verify feature diversity computation on orthogonal vs identical embeddings."""
    # 1. Two orthogonal vectors -> cosine distance = 1.0
    ortho_feats = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    res_ortho = verifier.compute_pairwise_cosine_diversity(ortho_feats)
    assert res_ortho["passed"] is True
    assert res_ortho["mean_pairwise_distance"] == pytest.approx(1.0, abs=1e-5)

    # 2. Identical vectors -> cosine distance = 0.0
    ident_feats = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    res_ident = verifier.compute_pairwise_cosine_diversity(ident_feats)
    assert res_ident["passed"] is False
    assert res_ident["mean_pairwise_distance"] == pytest.approx(0.0, abs=1e-5)


def test_tier1_patch_seam_gradient_ratio_smooth_vs_spiked():
    """Verify ViT patch-seam gradient ratio detects boundary discontinuities."""
    B, H, W = 4, 64, 64
    # 1. Smooth gradient image (no boundary artifacts)
    grid_y = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(B, 3, H, W)
    grid_x = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(B, 3, H, W)
    smooth_img = (grid_y + grid_x) * 0.5
    res_smooth = verifier.compute_patch_seam_gradient_ratio(smooth_img)
    assert res_smooth["passed"] is True
    assert res_smooth["patch_seam_ratio"] <= verifier.PATCH_SEAM_RATIO_THRESHOLD

    # 2. Image with harsh 16px patch spikes along seams
    spiked_img = smooth_img.clone()
    y_coords = torch.arange(H)
    seams = (y_coords % 16 == 0)
    spiked_img[:, :, seams, :] += 5.0
    res_spiked = verifier.compute_patch_seam_gradient_ratio(spiked_img)
    assert res_spiked["passed"] is False
    assert res_spiked["patch_seam_ratio"] > verifier.PATCH_SEAM_RATIO_THRESHOLD


def test_tier1_shift_sensitivity_invariance():
    """Verify shift sensitivity detects excessive probability changes under translation."""
    N, K = 64, 5
    orig_p = torch.zeros(N, K)
    orig_p[:, 0] = 0.90
    orig_p[:, 1:] = 0.025

    # Small shift change (delta = 0.01 < 0.12)
    shifted_small = orig_p.clone()
    shifted_small[:, 0] = 0.89
    shifted_small[:, 1] = 0.035
    res_small = verifier.compute_shift_sensitivity(orig_p, shifted_small, target_indices=[0])
    assert res_small["passed"] is True
    assert res_small["mean_shift_sensitivity"] == pytest.approx(0.01, abs=1e-5)

    # Large shift change (delta = 0.30 >= 0.12)
    shifted_large = orig_p.clone()
    shifted_large[:, 0] = 0.60
    shifted_large[:, 1] = 0.325
    res_large = verifier.compute_shift_sensitivity(orig_p, shifted_large, target_indices=[0])
    assert res_large["passed"] is False
    assert res_large["mean_shift_sensitivity"] == pytest.approx(0.30, abs=1e-5)


# ==============================================================================
# Tier 2: Boundary Value Analysis & Degenerate Tensors
# ==============================================================================

def test_tier2_boundary_sample_counts():
    """Verify sample count validation: exactly N=64 is valid; N=63 is rejected."""
    # N=63 raises ValueError
    fixtures_63 = verifier.generate_synthetic_test_batch(batch_size=63, scenario="pass")
    with pytest.raises(ValueError, match="at least 64 samples"):
        verifier.evaluate_acceptance(
            images=fixtures_63["images"],
            features=fixtures_63["features"],
            in_dist_probs=fixtures_63["in_distribution_probs"],
            audit_probs=fixtures_63["audit_probs"],
        )

    # N=64 passes
    fixtures_64 = verifier.generate_synthetic_test_batch(batch_size=64, scenario="pass")
    report_64 = verifier.evaluate_acceptance(
        images=fixtures_64["images"],
        features=fixtures_64["features"],
        in_dist_probs=fixtures_64["in_distribution_probs"],
        audit_probs=fixtures_64["audit_probs"],
    )
    assert report_64["status"] == "PASS"
    assert report_64["sample_count"] == 64


def test_tier2_probability_threshold_boundaries():
    """Test behavior at exact numerical threshold boundaries."""
    N = 64
    # 1. Target mode prob boundary: exactly 0.70 passes, 0.699 fails
    probs_pass = {
        q["aspect"]: torch.full((N, len(q["options"])), 0.0)
        for q in verifier.IN_DISTRIBUTION_QUERIES
    }
    for q in verifier.IN_DISTRIBUTION_QUERIES:
        aspect = q["aspect"]
        probs_pass[aspect][:, q["valid_target_modes"][0]] = 0.70
        probs_pass[aspect][:, q["noise_index"]] = 0.05
        probs_pass[aspect][:, q["unknown_index"]] = 0.08
    res_at_boundary = verifier.compute_in_distribution_satisfaction(probs_pass)
    assert res_at_boundary["passed_target"] is True
    assert res_at_boundary["passed_noise"] is True
    assert res_at_boundary["passed_unknown"] is True
    assert res_at_boundary["passed"] is True

    # Slightly below target threshold: 0.699
    probs_fail = {
        q["aspect"]: torch.full((N, len(q["options"])), 0.0)
        for q in verifier.IN_DISTRIBUTION_QUERIES
    }
    for q in verifier.IN_DISTRIBUTION_QUERIES:
        aspect = q["aspect"]
        probs_fail[aspect][:, q["valid_target_modes"][0]] = 0.699
        probs_fail[aspect][:, q["noise_index"]] = 0.05
        probs_fail[aspect][:, q["unknown_index"]] = 0.08
    res_below_boundary = verifier.compute_in_distribution_satisfaction(probs_fail)
    assert res_below_boundary["passed_target"] is False
    assert res_below_boundary["passed"] is False


def test_tier2_degenerate_and_invalid_inputs():
    """Verify clean error handling on degenerate or invalid tensor inputs."""
    # 1. Flat constant image (gradient is 0 everywhere)
    flat_img = torch.zeros(4, 3, 64, 64)
    res_flat = verifier.compute_patch_seam_gradient_ratio(flat_img)
    assert res_flat["passed"] is True  # 0 gradient does not have seam spikes

    # 2. Non-4D image tensor raises ValueError
    with pytest.raises(ValueError, match="4D tensor"):
        verifier.compute_patch_seam_gradient_ratio(torch.randn(3, 64, 64))

    # 3. Non-RGB image tensor raises ValueError
    with pytest.raises(ValueError, match="4D tensor"):
        verifier.compute_patch_seam_gradient_ratio(torch.randn(4, 1, 64, 64))

    # 4. M < 2 modes in entropy calculation raises ValueError
    with pytest.raises(ValueError, match="at least M >= 2"):
        verifier.compute_mode_marginal_entropy(torch.ones(10, 1))

    # 5. Features N < 2 raises ValueError
    with pytest.raises(ValueError, match="N >= 2"):
        verifier.compute_pairwise_cosine_diversity(torch.randn(1, 2560))


# ==============================================================================
# Tier 3: Combinations & Anti-Goodharting Evaluation
# ==============================================================================

def test_tier3_spatial_and_cyclic_transformations():
    """Verify spatial transforms and cyclic option rotation helpers."""
    images = torch.randn(8, 3, 64, 64)
    transforms = verifier.apply_spatial_transformations(images, shift=(4, 4), crop_scale=0.9)
    assert "original" in transforms
    assert "shifted" in transforms
    assert "crop" in transforms
    assert transforms["shifted"].shape == images.shape
    assert transforms["crop"].shape == images.shape
    # Ensure shifted image is not identical to original
    assert not torch.equal(transforms["shifted"], images)

    # Test cyclic option rotations
    options = ["apple", "citrus", "banana", "noise", "unknown"]
    rotations = verifier.apply_cyclic_option_rotations(options, n_rotations=4)
    assert len(rotations) == 4
    for orig_indices, rot_opts in rotations:
        assert len(rot_opts) == len(options)
        # Unknown remains at the end
        assert rot_opts[-1] == "unknown"
        assert orig_indices[-1] == 4


def test_tier3_failure_scenario_classification():
    """Verify distinct failure scenarios produce precise diagnosis flags."""
    # 1. Mode Collapse scenario: fails mode_entropy & feature_diversity, but passes seam coherence
    coll_fix = verifier.generate_synthetic_test_batch(batch_size=64, scenario="collapse")
    coll_rep = verifier.evaluate_acceptance(
        images=coll_fix["images"],
        features=coll_fix["features"],
        in_dist_probs=coll_fix["in_distribution_probs"],
        audit_probs=coll_fix["audit_probs"],
    )
    assert coll_rep["status"] == "FAIL"
    assert coll_rep["criteria"]["multi_mode_entropy"] is False
    assert coll_rep["criteria"]["feature_diversity"] is False
    assert coll_rep["criteria"]["patch_seam_coherence"] is True

    # 2. Patch Static scenario: fails patch_seam_coherence
    static_fix = verifier.generate_synthetic_test_batch(batch_size=64, scenario="patch_static")
    static_rep = verifier.evaluate_acceptance(
        images=static_fix["images"],
        features=static_fix["features"],
        in_dist_probs=static_fix["in_distribution_probs"],
        audit_probs=static_fix["audit_probs"],
    )
    assert static_rep["status"] == "FAIL"
    assert static_rep["criteria"]["patch_seam_coherence"] is False

    # 3. Noise scenario: fails in_distribution & held_out_audit
    noise_fix = verifier.generate_synthetic_test_batch(batch_size=64, scenario="noise")
    noise_rep = verifier.evaluate_acceptance(
        images=noise_fix["images"],
        features=noise_fix["features"],
        in_dist_probs=noise_fix["in_distribution_probs"],
        audit_probs=noise_fix["audit_probs"],
    )
    assert noise_rep["status"] == "FAIL"
    assert noise_rep["criteria"]["in_distribution_joint_satisfaction"] is False
    assert noise_rep["criteria"]["held_out_audit_accuracy"] is False


# ==============================================================================
# Tier 4: Turnkey CLI Execution & Mock Checkpoint Integration
# ==============================================================================

def test_tier4_cli_standalone_synthetic_pass():
    """Verify CLI standalone run with --synthetic returns exit code 0 and valid JSON."""
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--samples", "64",
        "--synthetic",
        "--scenario", "pass",
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"CLI failed: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert data["status"] == "PASS"
    assert data["sample_count"] == 64
    assert all(data["criteria"].values())


def test_tier4_cli_synthetic_failure_exit_code():
    """Verify CLI exits with code 1 on failed acceptance criteria."""
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--samples", "64",
        "--synthetic",
        "--scenario", "collapse",
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 1
    data = json.loads(proc.stdout)
    assert data["status"] == "FAIL"
    assert data["criteria"]["multi_mode_entropy"] is False


def test_tier4_cli_boundary_sample_count_rejection():
    """Verify CLI exits with code 2 when sample count < 64."""
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--samples", "63",
        "--synthetic",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 2
    assert "samples must be >= 64" in proc.stderr


def test_tier4_cli_output_dir_report_generation(tmp_path):
    """Verify --output-dir saves imajev_acceptance_report.json."""
    out_dir = tmp_path / "eval_out"
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--samples", "64",
        "--synthetic",
        "--output-dir", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    report_file = out_dir / "imajev_acceptance_report.json"
    assert report_file.is_file()
    saved = json.loads(report_file.read_text(encoding="utf-8"))
    assert saved["status"] == "PASS"
    assert saved["sample_count"] == 64


def _build_mock_generator() -> nn.Module:
    """Build a pure torch.nn module generator producing [B, 3, 64, 64] image tensors."""
    conv = nn.Conv2d(64, 3, kernel_size=1)
    with torch.no_grad():
        conv.weight.fill_(0.01)
        conv.bias.fill_(0.1)
    return nn.Sequential(
        nn.Unflatten(1, (64, 1, 1)),
        nn.Upsample(size=(64, 64), mode="nearest"),
        conv,
        nn.Tanh(),
    )


def test_tier4_mock_checkpoint_loading_and_evaluation(tmp_path):
    """Verify loading and evaluation of a mock generator checkpoint file."""
    import os
    ckpt_path = tmp_path / "mock_checkpoint.pt"
    generator = _build_mock_generator()
    torch.save({"generator": generator}, ckpt_path)

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--checkpoint", str(ckpt_path),
        "--samples", "64",
        "--device", "cpu",
        "--json",
    ]
    env = {**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}"}
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert proc.returncode == 0, f"Execution failed: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert data["sample_count"] == 64
    assert data["status"] == "PASS"
    assert data["criteria"]["patch_seam_coherence"] is True
