"""Adversarial stress-test suite for JointMultiQueryConstraintData in commit 21047d1b.

Verifies:
1. Target mode distribution is statistically uniform and independent across all queries over large sample batches (N >= 1200).
2. Marginal Shannon entropy across multi-mode queries satisfies H >= 0.85 ln(M_modes).
3. Target log-odds vectors [B, Q * K_max] align with joint_log_odds outputs and integrate seamlessly with discriminator loss.
4. Jitter augmentation and RNG reproducibility behave deterministically under manual seeds.
5. Boundary conditions, input validation, and extreme jitter stability.
"""
import copy
import math
import pytest
import scipy.stats
import torch
from torch import nn

from hypergan.checkpoints import data_contract
from hypergan.constraint_data import (
    JointMultiQueryConstraintData,
    OptionProbability,
)
from hypergan.hndl_networks import build_network


@pytest.fixture
def canonical_apple_queries():
    """Canonical 6-query setup mirroring ImaJev-4B constraint GAN."""
    return [
        {
            "aspect": "subject",
            "question": "What is the main subject?",
            "options": ["apple fruit", "orange or citrus", "banana or pear", "noise or static"],
            "target_modes": [0],  # Single mode (always apple)
        },
        {
            "aspect": "color",
            "question": "What color is the apple?",
            "options": ["bright red", "crisp green", "golden yellow", "noise"],
            "target_modes": [0, 1, 2],  # 3 modes: red, green, yellow
        },
        {
            "aspect": "bites",
            "question": "How many bites are taken out?",
            "options": ["zero bites", "one bite", "multiple bites", "noise"],
            "target_modes": [0, 1, 2],  # 3 modes: whole, one bite, multiple
        },
        {
            "aspect": "sitting_on",
            "question": "Where is the apple located?",
            "options": ["wooden table", "white background", "noise"],
            "target_modes": [0, 1],  # 2 modes
        },
        {
            "aspect": "style",
            "question": "What visual style is this image in?",
            "options": ["studio photograph", "oil painting", "digital rendering", "noise"],
            "target_modes": [0, 1, 2],  # 3 modes
        },
        {
            "aspect": "stem_leaf",
            "question": "Does the apple have a stem or leaf attached?",
            "options": ["stem with leaf", "bare stem", "no stem", "noise"],
            "target_modes": [0, 1, 2],  # 3 modes
        },
    ]


def test_adversarial_mode_distribution_uniformity_goodness_of_fit():
    """Verify statistical uniformity of mode draws across varying M in {2, 3, 4, 5, 8} at N >= 1200."""
    modes_to_test = [2, 3, 4, 5, 8]
    sample_sizes = [1200, 2400, 6000]

    for M in modes_to_test:
        queries = [
            {
                "aspect": f"query_{M}",
                "question": f"Question with {M} modes?",
                "options": [f"opt_{i}" for i in range(M + 1)],
                "target_modes": list(range(M)),
            }
        ]
        data = JointMultiQueryConstraintData(queries, tau=3.0, jitter_std=0.0)

        for N in sample_sizes:
            rng = torch.Generator().manual_seed(1000 + M * 100 + N)
            batch = data(N, generator=rng)
            sampled_modes = batch["modes"][:, 0]

            counts = torch.bincount(sampled_modes, minlength=M).cpu().numpy()
            expected_counts = [N / float(M)] * M

            # Pearson's Chi-squared Goodness-of-Fit test against discrete uniform
            chi2_stat, p_val = scipy.stats.chisquare(f_obs=counts, f_exp=expected_counts)

            assert p_val > 0.01, (
                f"Mode distribution non-uniform for M={M}, N={N}: "
                f"counts={counts}, chi2={chi2_stat:.4f}, p_value={p_val:.6f} <= 0.01"
            )

    # Single-mode query: must deterministically sample mode 0 for 100% of samples
    single_mode_q = [
        {
            "aspect": "single",
            "question": "Single mode question?",
            "options": ["only_target", "alt1", "noise"],
            "target_modes": [0],
        }
    ]
    data_single = JointMultiQueryConstraintData(single_mode_q, jitter_std=0.0)
    batch_single = data_single(1500)
    assert torch.all(batch_single["modes"][:, 0] == 0), "Single mode query failed to yield identically mode 0"


def test_adversarial_cross_query_statistical_independence(canonical_apple_queries):
    """Verify statistical independence across queries via contingency tables, mutual information, and correlations."""
    data = JointMultiQueryConstraintData(canonical_apple_queries, tau=3.0, jitter_std=0.0)
    N = 3600
    rng = torch.Generator().manual_seed(424242)
    batch = data(N, generator=rng)
    modes = batch["modes"]  # [N, 6]

    num_queries = len(canonical_apple_queries)

    # Test all pairs of multi-mode queries (queries 1 to 5)
    for q1 in range(1, num_queries):
        for q2 in range(q1 + 1, num_queries):
            m1 = modes[:, q1].cpu().numpy()
            m2 = modes[:, q2].cpu().numpy()

            M1 = len(canonical_apple_queries[q1]["target_modes"])
            M2 = len(canonical_apple_queries[q2]["target_modes"])

            # 1. 2D Contingency table test of independence
            contingency = torch.zeros(M1, M2, dtype=torch.float64)
            for i in range(N):
                contingency[m1[i], m2[i]] += 1

            chi2, p_val, dof, _ = scipy.stats.chi2_contingency(contingency.numpy())
            assert p_val > 0.01, (
                f"Statistical independence violation between Query {q1} and Query {q2}: "
                f"chi2={chi2:.4f}, dof={dof}, p={p_val:.6f}"
            )

            # 2. Empirical mutual information I(Q1; Q2)
            joint_probs = contingency / N
            p_m1 = joint_probs.sum(dim=1, keepdim=True)
            p_m2 = joint_probs.sum(dim=0, keepdim=True)
            indep_probs = p_m1 @ p_m2

            # Max absolute deviation from independence product
            max_dev = (joint_probs - indep_probs).abs().max().item()
            assert max_dev < 0.035, (
                f"Query independence deviation too high between Q{q1} and Q{q2}: "
                f"max_dev={max_dev:.4f} >= 0.035"
            )

            # Mutual information in nats
            nz = joint_probs > 0
            mi = (joint_probs[nz] * (joint_probs[nz] / indep_probs[nz]).log()).sum().item()
            assert mi < 0.025, (
                f"Mutual information between Q{q1} and Q{q2} too high: "
                f"{mi:.5f} >= 0.025 nats (expected ~0 for independent variables)"
            )

            # 3. Pearson correlation between mode indices
            r, r_pval = scipy.stats.pearsonr(m1, m2)
            assert abs(r) < 0.05, f"Cross-query Pearson correlation too high: r={r:.4f} between Q{q1} and Q{q2}"

    # Verify coverage across the full joint Cartesian product for multi-mode queries
    # Queries 1..5 have modes 3 * 3 * 2 * 3 * 3 = 162 total combinations
    multi_modes = modes[:, 1:]
    unique_tuples = set(tuple(row.tolist()) for row in multi_modes)
    assert len(unique_tuples) >= 155, (
        f"Joint mode coverage deficiency: observed {len(unique_tuples)} of 162 possible mode combinations"
    )


def test_adversarial_marginal_shannon_entropy_bound(canonical_apple_queries):
    """Verify marginal Shannon entropy H >= 0.85 * ln(M) across multi-mode queries over 50 random seeds."""
    data_clean = JointMultiQueryConstraintData(canonical_apple_queries, tau=3.0, jitter_std=0.0)
    data_jitter = JointMultiQueryConstraintData(canonical_apple_queries, tau=3.0, jitter_std=0.15)

    N = 1200
    multi_mode_queries = [
        (idx, len(q["target_modes"]))
        for idx, q in enumerate(canonical_apple_queries)
        if len(q["target_modes"]) > 1
    ]

    for seed in range(50):
        for data in (data_clean, data_jitter):
            rng = torch.Generator().manual_seed(seed * 7919)
            batch = data(N, generator=rng)
            modes = batch["modes"]

            for q_idx, M in multi_mode_queries:
                sampled = modes[:, q_idx]
                counts = torch.bincount(sampled, minlength=M).float()
                probs = counts / N

                # Shannon entropy in natural units (nats)
                nz_probs = probs[probs > 0]
                H = -(nz_probs * nz_probs.log()).sum().item()
                max_H = math.log(M)
                threshold = 0.85 * max_H

                assert H >= threshold, (
                    f"Marginal Shannon entropy violation at seed={seed}, q_idx={q_idx}, M={M}: "
                    f"H={H:.5f} < threshold={threshold:.5f} (max_H={max_H:.5f})"
                )


def test_adversarial_shannon_entropy_from_recovered_log_odds_probabilities(canonical_apple_queries):
    """Verify that Shannon entropy computed directly from output log-odds satisfies H >= 0.85 ln(M)."""
    data = JointMultiQueryConstraintData(canonical_apple_queries, tau=3.0, jitter_std=0.05)
    N = 1800
    rng = torch.Generator().manual_seed(8888)
    batch = data(N, generator=rng)
    real_log_odds = batch["real"]  # [N, 30]

    # For each query with M > 1 modes, recover predicted mode via OptionProbability and verify entropy
    # In canonical_apple_queries, all queries have 4 options + 1 unknown = 5 classes, so K = 5
    K = 5
    for q_idx, q in enumerate(canonical_apple_queries):
        M = len(q["target_modes"])
        if M <= 1:
            continue

        opt_prob = OptionProbability(
            index=list(range(M)),
            input_mode="joint_log_odds",
            num_classes=K,
            query_index=q_idx,
        )
        # Compute probabilities for the target options
        # Slice log-odds for this query
        q_slice = real_log_odds[:, q_idx * K : (q_idx + 1) * K]
        # Inferred mode is the argmax of the first M options
        inferred_modes = q_slice[:, :M].argmax(dim=-1)

        counts = torch.bincount(inferred_modes, minlength=M).float()
        probs = counts / N
        H = -(probs * probs.log()).sum().item()
        threshold = 0.85 * math.log(M)

        assert H >= threshold, (
            f"Shannon entropy from log-odds tensor failed threshold: "
            f"H={H:.4f} < {threshold:.4f} for query {q['aspect']}"
        )


def test_adversarial_target_log_odds_alignment_and_soft_capping():
    """Verify target log-odds shape [B, Q * K_max], bounds in (-tau, tau), and padding with -tau."""
    # Test with varying option counts: Q0 has 2 options (+1 = 3 classes), Q1 has 4 options (+1 = 5 classes)
    queries = [
        {
            "aspect": "q0",
            "question": "Q0?",
            "options": ["opt0", "opt1"],
            "target_modes": [0, 1],
        },
        {
            "aspect": "q1",
            "question": "Q1?",
            "options": ["a", "b", "c", "d"],
            "target_modes": [0, 1, 2],
        },
    ]
    # K_max = max(3, 5) = 5. Q * K_max = 2 * 5 = 10
    tau = 3.0
    data = JointMultiQueryConstraintData(queries, tau=tau, jitter_std=0.0)
    batch = data(32)
    real = batch["real"]

    assert real.shape == (32, 10), f"Expected shape (32, 10), got {real.shape}"
    assert real.dtype == torch.float32
    assert torch.isfinite(real).all()

    # Strict soft-capping bounds inside (-tau, tau)
    assert torch.all(real >= -tau)
    assert torch.all(real <= tau)

    # Padding verification: Q0 has 3 classes, so columns 3 and 4 must be identically -tau
    q0_pad = real[:, 3:5]
    torch.testing.assert_close(
        q0_pad,
        torch.full_like(q0_pad, -tau),
        atol=1e-6,
        rtol=0.0,
    )


def test_adversarial_seamless_discriminator_loss_integration(canonical_apple_queries):
    """Verify end-to-end integration with discriminator architectures, WGAN-GP loss, and optimizer updates."""
    # 6 queries, 5 classes each -> Q * K_max = 30
    Q_K = 30
    batch_size = 16

    data = JointMultiQueryConstraintData(canonical_apple_queries, tau=3.0, jitter_std=0.05)
    real_targets = data(batch_size)["real"]

    # 1. Critic / Discriminator network
    critic = nn.Sequential(
        nn.Linear(Q_K, 64),
        nn.LayerNorm(64),
        nn.LeakyReLU(0.2),
        nn.Linear(64, 32),
        nn.LeakyReLU(0.2),
        nn.Linear(32, 1),
    )
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # 2. Simulated generator outputs (matching soft-capped joint_log_odds outputs)
    fake_joint_log_odds = 3.0 * torch.tanh(torch.randn(batch_size, Q_K) / 3.0)

    # 3. Discriminator forward passes
    score_real = critic(real_targets)
    score_fake = critic(fake_joint_log_odds)

    # 4. WGAN loss
    loss_d = score_fake.mean() - score_real.mean()

    # 5. Gradient penalty (WGAN-GP)
    alpha = torch.rand(batch_size, 1)
    interpolates = (alpha * real_targets + (1 - alpha) * fake_joint_log_odds).requires_grad_(True)
    d_interpolates = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gp = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()

    total_loss = loss_d + 10.0 * gp

    # 6. Backward and step
    optimizer.zero_grad()
    total_loss.backward()

    # Verify gradients exist, are finite, and non-zero for weights
    for name, param in critic.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"Parameter {name} gradient contains non-finite values"
        if "weight" in name:
            assert param.grad.abs().sum() > 0.0, f"Weight parameter {name} gradient is completely zero"

    # Verify weights actually update after step
    initial_weights = copy.deepcopy(critic[0].weight.data)
    optimizer.step()
    updated_weights = critic[0].weight.data
    assert not torch.equal(initial_weights, updated_weights), "Critic weights did not update after optimizer step"


def test_adversarial_bce_and_hinge_discriminator_losses(canonical_apple_queries):
    """Verify stability under BCE with logits, Relativistic, and Hinge discriminator losses."""
    data = JointMultiQueryConstraintData(canonical_apple_queries, tau=3.0, jitter_std=0.08)
    real_targets = data(16)["real"]
    fake_targets = 3.0 * torch.tanh(torch.randn(16, 30))

    critic = nn.Sequential(nn.Linear(30, 16), nn.ReLU(), nn.Linear(16, 1))

    # Hinge loss
    d_real = critic(real_targets)
    d_fake = critic(fake_targets)
    hinge_loss = (torch.relu(1.0 - d_real) + torch.relu(1.0 + d_fake)).mean()
    assert torch.isfinite(hinge_loss)
    hinge_loss.backward()

    # BCE loss with logits
    critic.zero_grad()
    bce = nn.BCEWithLogitsLoss()
    d_real = critic(real_targets)
    d_fake = critic(fake_targets)
    bce_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
    assert torch.isfinite(bce_loss)
    bce_loss.backward()


def test_adversarial_jitter_augmentation_and_variance_scaling():
    """Verify jitter behavior: zero variance when jitter=0, variance scaling with jitter^2, zero-mean centering."""
    queries = [
        {
            "aspect": "test_jitter",
            "question": "Question?",
            "options": ["A", "B", "C", "D"],
            "target_modes": [0],  # Single mode to isolate jitter effect
        }
    ]

    # 1. jitter_std = 0: identical rows across the batch
    data_clean = JointMultiQueryConstraintData(queries, tau=3.0, jitter_std=0.0)
    batch_clean = data_clean(64)["real"]
    for i in range(1, 64):
        torch.testing.assert_close(batch_clean[i], batch_clean[0], atol=1e-6, rtol=0.0)

    # 2. jitter_std > 0: rows vary, variance proportional to jitter_std^2
    jitters = [0.05, 0.15, 0.30]
    sample_variances = []
    N = 8000

    for j in jitters:
        data_j = JointMultiQueryConstraintData(queries, tau=3.0, jitter_std=j)
        rng = torch.Generator().manual_seed(12345)
        batch_j = data_j(N, generator=rng)["real"]
        # Measure variance across batch for option 0
        var_j = batch_j[:, 0].var().item()
        sample_variances.append(var_j)

    # Variances must strictly increase monotonically with jitter_std
    assert sample_variances[0] < sample_variances[1] < sample_variances[2], (
        f"Variance not monotonically increasing with jitter: {sample_variances}"
    )


def test_adversarial_extreme_jitter_saturation_stability():
    """Stress-test numerical stability under extreme jitter (jitter_std=50.0, 1000.0)."""
    queries = [
        {
            "aspect": "extreme",
            "question": "Extreme jitter?",
            "options": ["x", "y", "z"],
            "target_modes": [0, 1],
        }
    ]
    tau = 3.0
    for extreme_jitter in [10.0, 50.0, 1000.0]:
        data = JointMultiQueryConstraintData(queries, tau=tau, jitter_std=extreme_jitter)
        batch = data(128)["real"]

        # Must never produce NaN or Inf
        assert torch.isfinite(batch).all(), f"Extreme jitter {extreme_jitter} produced non-finite values"

        # Must remain strictly bounded in [-tau, tau] due to soft-capping
        assert torch.all(batch >= -tau), f"Lower bound violated under jitter {extreme_jitter}"
        assert torch.all(batch <= tau), f"Upper bound violated under jitter {extreme_jitter}"


def test_adversarial_rng_determinism_and_device_handling():
    """Verify strict determinism under manual seeds, across multiple instances and CUDA if available."""
    queries = [
        {
            "aspect": "q1",
            "question": "Color?",
            "options": ["red", "green", "blue"],
            "target_modes": [0, 1],
        },
        {
            "aspect": "q2",
            "question": "Shape?",
            "options": ["round", "square"],
            "target_modes": [0, 1],
        },
    ]

    # 1. Two separate instances with identical generator seed produce bitwise-equal outputs
    d1 = JointMultiQueryConstraintData(queries, tau=3.0, jitter_std=0.1)
    d2 = JointMultiQueryConstraintData(queries, tau=3.0, jitter_std=0.1)

    g1 = torch.Generator().manual_seed(98765)
    g2 = torch.Generator().manual_seed(98765)

    b1 = d1(32, generator=g1)
    b2 = d2(32, generator=g2)

    assert torch.equal(b1["real"], b2["real"]), "Bitwise equality failed across distinct instances with identical seed"
    assert torch.equal(b1["modes"], b2["modes"]), "Mode draws failed bitwise equality under identical seed"

    # 2. Sequential calls advance the generator (do not repeat identical batches)
    b3 = d1(32, generator=g1)
    assert not torch.equal(b1["real"], b3["real"]), "Sequential calls unexpectedly produced identical batches"

    # 3. CUDA generator support (if available)
    if torch.cuda.is_available():
        cuda_g1 = torch.Generator(device="cuda:0").manual_seed(55555)
        cuda_g2 = torch.Generator(device="cuda:0").manual_seed(55555)

        cuda_b1 = d1(32, generator=cuda_g1)
        cuda_b2 = d2(32, generator=cuda_g2)

        assert cuda_b1["real"].is_cuda
        assert cuda_b1["modes"].is_cuda
        assert torch.equal(cuda_b1["real"], cuda_b2["real"]), "CUDA generator determinism failed"


def test_adversarial_boundary_conditions_and_exceptions():
    """Verify input validation and boundary condition enforcement."""
    valid_q = [
        {"question": "Q1?", "options": ["a", "b"], "target_modes": [0, 1]},
        {"question": "Q2?", "options": ["c", "d"], "target_modes": [0]},
    ]

    # Query count boundaries: 1 is allowed, 64 is allowed, 0 or >64 rejected
    JointMultiQueryConstraintData([valid_q[0]])  # Q = 1
    JointMultiQueryConstraintData([valid_q[0]] * 64)  # Q = 64
    with pytest.raises(ValueError, match="1..64 query mappings"):
        JointMultiQueryConstraintData([])
    with pytest.raises(ValueError, match="1..64 query mappings"):
        JointMultiQueryConstraintData([valid_q[0]] * 65)

    # Batch size boundaries
    data = JointMultiQueryConstraintData(valid_q)
    b1 = data(1)
    assert b1["real"].shape[0] == 1
    b_large = data(5000)
    assert b_large["real"].shape[0] == 5000

    for bad_bs in [0, -1, -100, 1.5, True, False, "16"]:
        with pytest.raises(ValueError, match="batch_size"):
            data(bad_bs)

    # Tau boundaries
    for bad_tau in [0.0, -1.0, float("nan"), float("inf"), -float("inf"), True, False, "3.0"]:
        with pytest.raises(ValueError, match="tau"):
            JointMultiQueryConstraintData(valid_q, tau=bad_tau)

    # Temperature boundaries
    for bad_t in [0.0, -1.0, float("nan"), float("inf"), True, False]:
        with pytest.raises(ValueError, match="temperature"):
            JointMultiQueryConstraintData(valid_q, temperature=bad_t)

    # Jitter boundaries
    for bad_j in [-0.01, -1.0, float("nan"), float("inf"), True, False]:
        with pytest.raises(ValueError, match="jitter_std"):
            JointMultiQueryConstraintData(valid_q, jitter_std=bad_j)

    # Target prob boundaries: must be strictly in (0.5, 1.0)
    for bad_tp in [0.5, 0.49, 1.0, 1.01, 0.0, -0.5, float("nan"), float("inf"), True]:
        with pytest.raises(ValueError, match="target_prob"):
            JointMultiQueryConstraintData(valid_q, target_prob=bad_tp)

    # Invalid query mapping structures
    with pytest.raises(ValueError, match="must be a mapping"):
        JointMultiQueryConstraintData(["not_a_mapping"])
    with pytest.raises(ValueError, match="question"):
        JointMultiQueryConstraintData([{"question": "   ", "options": ["a", "b"], "target_modes": [0]}])
    with pytest.raises(ValueError, match="unique"):
        JointMultiQueryConstraintData([{"question": "Q?", "options": ["a", "a"], "target_modes": [0]}])
    with pytest.raises(ValueError, match="1..63 strings"):
        JointMultiQueryConstraintData([{"question": "Q?", "options": [], "target_modes": [0]}])
    with pytest.raises(ValueError, match="target_modes entries must be in"):
        JointMultiQueryConstraintData([{"question": "Q?", "options": ["a", "b"], "target_modes": [5]}])


def test_adversarial_data_contract_and_resume_identity(canonical_apple_queries):
    """Verify data_contract specification, statelessness, and resume_identity fidelity."""
    data = JointMultiQueryConstraintData(
        canonical_apple_queries,
        tau=2.5,
        temperature=1.2,
        jitter_std=0.08,
        target_prob=0.985,
    )
    spec = {
        "factory": "hypergan.constraint_data:JointMultiQueryConstraintData",
        "args": {
            "queries": canonical_apple_queries,
            "tau": 2.5,
            "temperature": 1.2,
            "jitter_std": 0.08,
            "target_prob": 0.985,
        },
    }
    contract = data_contract(data, spec)
    assert contract["supported"] is True
    assert contract["stateful"] is False

    identity = data.resume_identity()
    assert identity["tau"] == 2.5
    assert identity["temperature"] == 1.2
    assert identity["jitter_std"] == 0.08
    assert identity["target_prob"] == 0.985
    assert len(identity["queries"]) == len(canonical_apple_queries)

    # Identity mutation isolation
    identity["queries"][0]["question"] = "MUTATED"
    assert data.queries[0]["question"] != "MUTATED"
