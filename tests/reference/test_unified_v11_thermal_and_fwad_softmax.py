"""Regression tests for Unified-v11 multi-critic thermal policy and fwAD softmax penalties."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn
import torch.nn.functional as F

from hypergan.checkpoints import restore_trainer, trainer_state
from hypergan.config import resolve_config
from hypergan.cpu_workers import launch_cpu_workers
from hypergan.distributed_training import ReplicatedCPUTrainer
from hypergan.particlegan_runtime import (
    DualSafeSoftmaxMode,
    MULTI_CRITIC_THERMAL_POLICY,
    critic_penalty_context,
    invoke_critic_penalty,
)
from hypergan.training import ReferenceTrainer


# ---------------------------------------------------------------------------
# Importable synthetic fixtures
# ---------------------------------------------------------------------------

class DualBranchGenerator(nn.Module):
    """Generator producing a 4D vector: first 2 dims near 0 (matches real_easy),
    last 2 dims small (far from shifted real_hard)."""

    def __init__(self, z_dim: int = 4, out_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, out_dim),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.05)
            self.net[-1].bias.zero_()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SliceExtractor(nn.Module):
    """Parameter-free slice selector splitting a shared generator output for two critics."""

    def __init__(self, start: int = 0, end: int = 2):
        super().__init__()
        self.start = int(start)
        self.end = int(end)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.start:self.end]


class SimpleLinearCritic(nn.Module):
    def __init__(self, in_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, candidate: torch.Tensor) -> torch.Tensor:
        return self.net(candidate)


class TwoDomainThermalData:
    """Two-domain synthetic data: 'real' is near zero (matching generator init so
    primary critic cools fast), while 'real_hard' is shifted far away (keeping
    secondary critic hot)."""

    resume_stateless = True

    def __init__(self, easy_std: float = 0.02, hard_shift: float = 6.0):
        self.easy_std = float(easy_std)
        self.hard_shift = float(hard_shift)

    def __call__(self, batch_size: int, *, generator: torch.Generator) -> dict[str, torch.Tensor]:
        easy = self.easy_std * torch.randn(batch_size, 2, generator=generator)
        hard = self.hard_shift + 0.5 * torch.randn(batch_size, 2, generator=generator)
        return {"real": easy, "real_hard": hard}


class HopfieldAttentionCritic(nn.Module):
    """Small critic containing Hopfield softmax memory, masked SDPA, MultiheadAttention,
    softmin, and log_softmax — exercises every PyTorch fwAD softmax failure path."""

    def __init__(self, in_dim: int = 8, num_patterns: int = 5):
        super().__init__()
        self.mem = nn.Parameter(torch.randn(num_patterns, in_dim) * 0.5)
        self.qkv = nn.Linear(in_dim, in_dim * 3)
        self.mha = nn.MultiheadAttention(in_dim, num_heads=2, batch_first=True)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, candidate: torch.Tensor) -> torch.Tensor:
        # 1. Modern Hopfield retrieval via torch.softmax
        scores = candidate @ self.mem.T
        weights = torch.softmax(scores, dim=-1)
        h_hop = weights @ self.mem

        # 2. Masked scaled_dot_product_attention (2 tokens x 4 dims)
        q, k, v = self.qkv(h_hop).chunk(3, dim=-1)
        bsz = candidate.shape[0]
        half = candidate.shape[-1] // 2
        q2 = q.view(bsz, 2, half)
        k2 = k.view(bsz, 2, half)
        v2 = v.view(bsz, 2, half)
        causal_bias = torch.tensor([[0.0, float("-inf")], [0.0, 0.0]], dtype=candidate.dtype, device=candidate.device)
        h_sdpa = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=causal_bias).reshape(bsz, -1)

        # 3. nn.MultiheadAttention (both need_weights=False and need_weights=True)
        seq = torch.stack([h_hop, h_sdpa], dim=1)
        h_mha_sdpa, _ = self.mha(seq, seq, seq, need_weights=False)
        h_mha_sm, _ = self.mha(seq, seq, seq, need_weights=True)

        # 4. softmin, log_softmax, torch.special.softmax & torch.special.log_softmax
        h_softmin = F.softmin(h_mha_sdpa.mean(dim=1), dim=-1)
        h_logsm = F.log_softmax(h_mha_sm.mean(dim=1), dim=-1)
        h_spec_sm = torch.special.softmax(h_hop, dim=-1)
        h_spec_lsm = torch.special.log_softmax(h_sdpa, dim=-1)
        return self.head(h_hop + h_sdpa + h_softmin + h_logsm + h_spec_sm + h_spec_lsm)


class PlainMLPCritic(nn.Module):
    """Non-softmax MLP critic for verifying zero interference from DualSafeSoftmaxMode."""

    def __init__(self, in_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(self, candidate: torch.Tensor) -> torch.Tensor:
        return self.net(candidate)


class SimpleVectorData:
    resume_stateless = True

    def __init__(self, dim: int = 8):
        self.dim = int(dim)

    def __call__(self, batch_size: int, *, generator: torch.Generator) -> dict[str, torch.Tensor]:
        return {"real": torch.randn(batch_size, self.dim, generator=generator)}


class SimpleVectorGenerator(nn.Module):
    def __init__(self, z_dim: int = 4, out_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def _factory(cls: type) -> str:
    return f"{__name__}:{cls.__name__}"


def _two_critic_config(*, steps: int = 6, batch_size: int = 8, seed: int = 11) -> dict:
    return {
        "defaults": "particlegan:unified_v11",
        "data": {"factory": _factory(TwoDomainThermalData), "args": {"easy_std": 0.02, "hard_shift": 6.0}},
        "prior": {"kind": "mog", "fixed_sigma": 0.05, "args": {"num_particles": 16, "z_dim": 4}},
        "components": {
            "trunk": {
                "factory": _factory(DualBranchGenerator),
                "args": {"z_dim": 4, "out_dim": 4},
                "inputs": {"z": "latent"},
            },
            "generator": {
                "factory": _factory(SliceExtractor),
                "args": {"start": 0, "end": 2},
                "inputs": {"x": "components.trunk"},
            },
            "hard_branch": {
                "factory": _factory(SliceExtractor),
                "args": {"start": 2, "end": 4},
                "inputs": {"x": "components.trunk"},
            },
            "discriminator": {
                "factory": _factory(SimpleLinearCritic),
                "args": {"in_dim": 2},
                "inputs": {"candidate": "candidate"},
            },
            "aux_critic": {
                "factory": _factory(SimpleLinearCritic),
                "args": {"in_dim": 2},
                "inputs": {"candidate": "candidate"},
            },
        },
        "adversarial": {
            "id": "easy_adv",
            "weight": 1.0,
            "critic_weight": 1.0,
            "generator_weight": 0.5,
            "real": "batch.real",
            "fake": "generated",
        },
        "adversarial_terms": [
            {
                "id": "hard_adv",
                "component": "aux_critic",
                "weight": 1.0,
                "critic_weight": 1.0,
                "generator_weight": 1.0,
                "real": "batch.real_hard",
                "fake": "components.hard_branch",
                "inputs": {"candidate": "candidate"},
                "penalty": True,
                "penalty_coeff": 0.5,
            }
        ],
        "gradient_penalty": {"coeff": 0.5, "lazy_k": 1},
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "seed": seed,
            "device": "cpu",
            "initialization": "constructor",
        },
        "sampling": {"count": 4, "seed": 23},
        "metrics": {"every_steps": 1},
    }


# ---------------------------------------------------------------------------
# Issue 1: Multi-critic _thermal_scale policy & no silent weight drift
# ---------------------------------------------------------------------------

def test_two_critic_thermal_policy_prevents_silent_weight_drift():
    """When one critic wants to cool >30x faster than another (e.g. 7e-4 vs 2.6e-2),
    MultiCriticThermalCoordinator holds every critic and shared opt_g/prior/spread at
    the exact same synchronized scale s_shared = max_i s_i, preventing silent drift
    of relative generator/prior gradient weights."""
    config = resolve_config(_two_critic_config(steps=5))
    trainer = ReferenceTrainer(config)

    coord = trainer.thermal_coordinator
    assert coord.active is True
    assert coord.policy == MULTI_CRITIC_THERMAL_POLICY

    rec_primary = trainer.critic_optimizers["discriminator"].record
    rec_aux = trainer.critic_optimizers["aux_critic"].record
    assert rec_primary is not rec_aux
    assert trainer.opt_g._unified_v11_record is rec_primary
    assert trainer.prior._unified_v11_record is rec_primary

    # Simulate realistic asymmetric cooling rates: primary critic cools to 7e-4 while
    # secondary critic cools to 2.6e-2 (>37x raw divergence).
    orig_p = rec_primary.observe_t1c_thermal
    orig_a = rec_aux.observe_t1c_thermal

    def cool_primary(*args, **kwargs):
        orig_p(*args, **kwargs)
        rec_primary._thermal_scale = 7e-4
        return 7e-4

    def cool_aux(*args, **kwargs):
        orig_a(*args, **kwargs)
        rec_aux._thermal_scale = 2.6e-2
        return 2.6e-2

    rec_primary.observe_t1c_thermal = cool_primary
    rec_aux.observe_t1c_thermal = cool_aux

    # Spy on the _thermal_scale actually seen inside each critic's forward hook during G phase
    seen_scales: dict[str, list[float]] = {"discriminator": [], "aux_critic": []}
    hooks = []
    for name in ("discriminator", "aux_critic"):
        rec_n = trainer.critic_optimizers[name].record
        hooks.append(
            trainer.graph.models[name].register_forward_hook(
                lambda _m, _i, _o, _name=name, _rec=rec_n: seen_scales[_name].append(float(_rec._thermal_scale))
            )
        )

    try:
        rows = []
        for _ in range(3):
            row, _ = trainer.update()
            rows.append(row)
            # Both records MUST agree identically after every step (zero drift!)
            assert rec_primary._thermal_scale == rec_aux._thermal_scale
            assert rec_primary._thermal_scale == pytest.approx(2.6e-2, rel=0, abs=0)
            # Telemetry exposes both synchronized effective scale and raw per-critic scales
            assert row["critic/thermal_scale"] == pytest.approx(2.6e-2)
            assert row["critic/easy_adv/thermal_scale"] == pytest.approx(2.6e-2)
            assert row["critic/hard_adv/thermal_scale"] == pytest.approx(2.6e-2)
            assert row["critic/easy_adv/raw_thermal_scale"] == pytest.approx(7e-4)
            assert row["critic/hard_adv/raw_thermal_scale"] == pytest.approx(2.6e-2)
    finally:
        for h in hooks:
            h.remove()

    # Every forward call of discriminator and aux_critic in the same step saw identical scales!
    assert len(seen_scales["discriminator"]) == len(seen_scales["aux_critic"])
    assert len(seen_scales["discriminator"]) > 0
    for s_d, s_a in zip(seen_scales["discriminator"], seen_scales["aux_critic"]):
        assert s_d == s_a
        assert s_d / s_a == pytest.approx(1.0, rel=0, abs=0)

    # Confirm step 2+ actually ran at the coupled 2.6e-2 scale (not stuck at 1.0, not dropped to 7e-4)
    assert seen_scales["discriminator"][-1] == pytest.approx(2.6e-2)
    assert seen_scales["aux_critic"][-1] == pytest.approx(2.6e-2)

    # Verify directly that relative generator gradient contributions from the two critics
    # keep their exact configured ratio (no 37x silent muting of discriminator relative to aux_critic):
    from hypergan.objective_program import _bound_scores

    batch = trainer.batch()
    z, _ = trainer.prior.sample(config["training"]["batch_size"], generator=trainer.streams["prior"])
    context = trainer.graph.generate(z, batch, prior=trainer.prior)
    term_easy, term_hard = trainer.program.adversarial_terms

    # Measure gradients with synchronized records (both at 2.6e-2) vs baseline at 1.0:
    def _term_grad_norm(term, scale_override=None):
        if scale_override is not None:
            term.gan.record._thermal_scale = scale_override
        trainer.opt_g.zero_grad(set_to_none=True)
        _, _, r_sc, f_sc = _bound_scores(
            term, context, trainer.graph, "generator", term.generator_phase,
            first="fake", noise=((0, None), (0, None)),
        )
        loss = term.generator_weight * term.gan.g_loss(f_sc, r_sc)
        grads = torch.autograd.grad(loss, list(trainer.program.generator_parameters), retain_graph=True, allow_unused=True)
        sq = sum(float(g.detach().square().sum().item()) for g in grads if g is not None)
        return math.sqrt(sq)

    # Hold records without mutating EMAs during this read-only ratio check
    saved_p, saved_a = rec_primary._thermal_scale, rec_aux._thermal_scale
    with trainer.thermal_coordinator.step_scope():
        g_easy_sync = _term_grad_norm(term_easy)
        g_hard_sync = _term_grad_norm(term_hard)
    # Restore to baseline 1.0 to get the true uncooled configured weight ratio
    rec_primary._thermal_scale = 1.0
    rec_aux._thermal_scale = 1.0
    with trainer.thermal_coordinator.step_scope():
        g_easy_hot = _term_grad_norm(term_easy)
        g_hard_hot = _term_grad_norm(term_hard)
    rec_primary._thermal_scale = saved_p
    rec_aux._thermal_scale = saved_a

    assert g_easy_sync > 0.0 and g_hard_sync > 0.0
    assert g_easy_hot > 0.0 and g_hard_hot > 0.0
    ratio_sync = g_easy_sync / g_hard_sync
    ratio_hot = g_easy_hot / g_hard_hot
    # Synchronized thermal scaling scales both terms identically -> ratio drift is < 1% (not >30x!)
    assert ratio_sync == pytest.approx(ratio_hot, rel=1e-2)

    # Now verify reheating: if primary critic detects a shock (returns 1.0) while aux is at 2.6e-2,
    # both immediately reheat together to 1.0.
    def shock_primary(*args, **kwargs):
        orig_p(*args, **kwargs)
        rec_primary._thermal_scale = 1.0
        return 1.0

    rec_primary.observe_t1c_thermal = shock_primary
    row_reheat, _ = trainer.update()
    assert rec_primary._thermal_scale == 1.0
    assert rec_aux._thermal_scale == 1.0
    assert row_reheat["critic/thermal_scale"] == pytest.approx(1.0)
    assert row_reheat["critic/easy_adv/raw_thermal_scale"] == pytest.approx(1.0)
    assert row_reheat["critic/hard_adv/raw_thermal_scale"] == pytest.approx(2.6e-2)


def test_two_critic_thermal_bit_exact_checkpoint_resume_and_legacy_alignment():
    """Multi-critic thermal coordination introduces zero uncheckpointed state (bit-exact
    checkpoint/resume) and aligns mismatched _thermal_scale values when restoring."""
    config = resolve_config(_two_critic_config(steps=4, seed=31))
    continuous = ReferenceTrainer(config)
    for _ in range(4):
        row_cont, _ = continuous.update()

    split = ReferenceTrainer(config)
    for _ in range(2):
        _, last_batch = split.update()
    saved = copy.deepcopy(trainer_state(split, last_batch))

    resumed = ReferenceTrainer(config)
    restore_trainer(resumed, saved)
    for _ in range(2):
        row_res, _ = resumed.update()

    assert row_res["d_loss"] == row_cont["d_loss"]
    assert row_res["g_loss"] == row_cont["g_loss"]
    assert row_res["critic/thermal_scale"] == row_cont["critic/thermal_scale"]
    for name in ("discriminator", "aux_critic"):
        assert (
            resumed.critic_optimizers[name].record._thermal_scale
            == continuous.critic_optimizers[name].record._thermal_scale
        )

    # Legacy checkpoint with divergent thermal scales is immediately synchronized on restore
    divergent = copy.deepcopy(saved)
    divergent["optimizers"][1]["regularizer"]["record"]["unified_v11"]["thermal_scale"] = 7e-4
    divergent["optimizers"][2]["regularizer"]["record"]["unified_v11"]["thermal_scale"] = 2.6e-2
    repaired = ReferenceTrainer(config)
    restore_trainer(repaired, divergent)
    assert repaired.critic_optimizers["discriminator"].record._thermal_scale == pytest.approx(2.6e-2)
    assert repaired.critic_optimizers["aux_critic"].record._thermal_scale == pytest.approx(2.6e-2)


def _replicated_two_critic_thermal_worker(rank: int, world_size: int, config: dict, shared_dir: str) -> None:
    trainer = ReplicatedCPUTrainer(config, world_size=world_size)
    rec_p = trainer.critic_optimizers["discriminator"].record
    rec_a = trainer.critic_optimizers["aux_critic"].record
    orig_p = rec_p.observe_t1c_thermal
    orig_a = rec_a.observe_t1c_thermal

    def cool_p(*args, **kwargs):
        orig_p(*args, **kwargs)
        rec_p._thermal_scale = 8e-4
        return 8e-4

    def cool_a(*args, **kwargs):
        orig_a(*args, **kwargs)
        rec_a._thermal_scale = 3e-2
        return 3e-2

    rec_p.observe_t1c_thermal = cool_p
    rec_a.observe_t1c_thermal = cool_a

    rows = []
    for _ in range(2):
        row, _ = trainer.update()
        rows.append(row)
    trainer._assert_replicas("two-critic thermal replicated")
    payload = {
        "rank": rank,
        "p_scale": float(rec_p._thermal_scale),
        "a_scale": float(rec_a._thermal_scale),
        "digest": trainer._replica_digest(),
        "rows": rows,
    }
    (Path(shared_dir) / f"rank-{rank}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_two_critic_thermal_replicated_cpu_agreement(tmp_path):
    """Two-rank ReplicatedCPUTrainer maintains identical synchronized thermal scales
    across both critics and all ranks without breaking replica digest checks."""
    config = resolve_config(_two_critic_config(steps=3, batch_size=8, seed=41))
    launch_cpu_workers(
        _replicated_two_critic_thermal_worker,
        world_size=2,
        args=(config, str(tmp_path)),
        timeout=60,
        collective_timeout=20,
    )
    results = [
        json.loads((tmp_path / f"rank-{r}.json").read_text(encoding="utf-8"))
        for r in range(2)
    ]
    assert results[0]["digest"] == results[1]["digest"]
    for res in results:
        assert res["p_scale"] == pytest.approx(3e-2)
        assert res["a_scale"] == pytest.approx(3e-2)
        for row in res["rows"]:
            assert row["critic/thermal_scale"] == pytest.approx(3e-2)
            assert row["critic/easy_adv/thermal_scale"] == pytest.approx(3e-2)
            assert row["critic/hard_adv/thermal_scale"] == pytest.approx(3e-2)
            assert row["critic/easy_adv/raw_thermal_scale"] == pytest.approx(8e-4)
            assert row["critic/hard_adv/raw_thermal_scale"] == pytest.approx(3e-2)


# ---------------------------------------------------------------------------
# Issue 2: PyTorch fwAD + torch.softmax backward (pytorch#162350)
# ---------------------------------------------------------------------------

def test_fwad_softmax_reproducer_and_unified_v11_training():
    """Reproduce the raw fwAD + torch.softmax backward inplace version-counter error
    and verify Unified-v11 ReferenceTrainer trains a softmax/Hopfield/attention critic
    end-to-end with finite losses and nonzero parameter gradients."""
    torch.manual_seed(0)
    raw_critic = HopfieldAttentionCritic(in_dim=8, num_patterns=5)
    x = torch.randn(4, 8)
    u = torch.randn(4, 8)

    # 1. Confirm raw fwAD + backward through softmax fails on affected PyTorch builds
    #    (or if an upstream PyTorch build fixes #162350, still verify DualSafeSoftmaxMode).
    raw_failed_with_exp_version_bug = False
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.MATH):
            with fwAD.dual_level():
                yd = raw_critic(fwAD.make_dual(x, u))
                _, jvp = fwAD.unpack_dual(yd)
        jvp.square().mean().backward()
    except RuntimeError as exc:
        assert "inplace operation" in str(exc) and "Exp" in str(exc)
        raw_failed_with_exp_version_bug = True
    assert raw_failed_with_exp_version_bug, (
        "Expected raw fwAD + softmax backward to fail with Exp inplace version error on this PyTorch build"
    )

    # 2. Verify full Unified-v11 ReferenceTrainer with HopfieldAttentionCritic succeeds
    cfg = resolve_config({
        "defaults": "particlegan:unified_v11",
        "data": {"factory": _factory(SimpleVectorData), "args": {"dim": 8}},
        "prior": {"kind": "mog", "fixed_sigma": 0.05, "args": {"num_particles": 16, "z_dim": 4}},
        "components": {
            "generator": {
                "factory": _factory(SimpleVectorGenerator),
                "args": {"z_dim": 4, "out_dim": 8},
                "inputs": {"z": "latent"},
            },
            "discriminator": {
                "factory": _factory(HopfieldAttentionCritic),
                "args": {"in_dim": 8, "num_patterns": 5},
                "inputs": {"candidate": "candidate"},
            },
        },
        "gradient_penalty": {"coeff": 1.0, "lazy_k": 1},
        "training": {"steps": 3, "batch_size": 8, "seed": 7, "device": "cpu", "initialization": "constructor"},
        "sampling": {"count": 4, "seed": 11},
        "metrics": {"every_steps": 1},
    })
    trainer = ReferenceTrainer(cfg)
    disc = trainer.graph.models["discriminator"]
    mem_before = disc.mem.detach().clone()
    qkv_before = disc.qkv.weight.detach().clone()
    mha_before = disc.mha.in_proj_weight.detach().clone()

    for _ in range(3):
        row, _ = trainer.update()
        assert math.isfinite(row["d_loss"])
        assert math.isfinite(row["g_loss"])
        assert math.isfinite(row["gradient_penalty"])
        assert row["gradient_penalty"] >= 0.0

    # Gradients actually flowed into softmax/Hopfield memory and attention weights
    assert (disc.mem.detach() - mem_before).abs().max().item() > 0.0
    assert (disc.qkv.weight.detach() - qkv_before).abs().max().item() > 0.0
    assert (disc.mha.in_proj_weight.detach() - mha_before).abs().max().item() > 0.0


def test_hndl_hopfield_softmax_critic_trains_under_unified_v11():
    """HNDL critic using native hopfield() and softmax() trains cleanly under Unified-v11."""
    cfg = resolve_config({
        "defaults": "particlegan:unified_v11",
        "data": {"factory": _factory(SimpleVectorData), "args": {"dim": 8}},
        "prior": {"kind": "mog", "fixed_sigma": 0.05, "args": {"num_particles": 16, "z_dim": 4}},
        "components": {
            "generator": {
                "factory": "hndl",
                "args": {
                    "source": "linear(16)\nleaky_relu(0.2)\nlinear()",
                    "input_shape": ["B", 4],
                    "output_shape": ["B", 8],
                },
                "inputs": {"x": "latent"},
            },
            "discriminator": {
                "factory": "hndl",
                "args": {
                    "source": "linear(8)\nhopfield(patterns=6)\nsoftmax()\nlinear()",
                    "input_shape": ["B", 8],
                    "output_shape": ["B", 1],
                },
                "inputs": {"x": "candidate"},
            },
        },
        "gradient_penalty": {"coeff": 1.0, "lazy_k": 1},
        "training": {"steps": 2, "batch_size": 8, "seed": 13, "device": "cpu"},
        "sampling": {"count": 4, "seed": 19},
        "metrics": {"every_steps": 1},
    })
    trainer = ReferenceTrainer(cfg)
    disc = trainer.graph.models["discriminator"]
    params_before = [p.detach().clone() for p in disc.parameters() if p.requires_grad]
    for _ in range(2):
        row, _ = trainer.update()
        assert math.isfinite(row["d_loss"])
        assert math.isfinite(row["gradient_penalty"])
    params_after = [p.detach() for p in disc.parameters() if p.requires_grad]
    assert any((a - b).abs().max().item() > 0.0 for a, b in zip(params_after, params_before))


def test_dual_safe_softmax_gradient_accuracy_vs_autograd_and_finite_differences():
    """Verify DualSafeSoftmaxMode matches exact double-backprop oracle to ~1e-13 and
    central finite differences to ~1e-6 on Hopfield/SDPA/MHA/softmax parameters, and
    remains bitwise identical to plain fwAD on non-softmax layers."""
    torch.manual_seed(42)
    critic = HopfieldAttentionCritic(in_dim=8, num_patterns=5).double()
    x = torch.randn(4, 8, dtype=torch.float64)
    u = torch.randn(4, 8, dtype=torch.float64)

    # 1. Exact double-backprop oracle (used in test verification only!)
    critic.zero_grad()
    x_req = x.clone().requires_grad_(True)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        y_ref = critic(x_req)
        g_x = torch.autograd.grad(y_ref.sum(), x_req, create_graph=True)[0]
        jvp_ref = (g_x * u).sum(dim=-1, keepdim=True)
        loss_ref = jvp_ref.square().mean() + 0.03 * y_ref.mean().square()
        loss_ref.backward()
    grads_ref = {n: p.grad.clone() for n, p in critic.named_parameters() if p.grad is not None}

    # 2. DualSafeSoftmaxMode + fwAD (zero autograd.grad calls)
    def eval_fwad_loss() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with critic_penalty_context():
            with fwAD.dual_level():
                yd = critic(fwAD.make_dual(x, u))
                y_val, jvp_val = fwAD.unpack_dual(yd)
        loss_val = jvp_val.square().mean() + 0.03 * y_val.mean().square()
        return y_val, jvp_val, loss_val

    critic.zero_grad()
    y_fw, jvp_fw, loss_fw = eval_fwad_loss()
    loss_fw.backward()
    grads_fw = {n: p.grad.clone() for n, p in critic.named_parameters() if p.grad is not None}

    torch.testing.assert_close(y_fw, y_ref, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(jvp_fw, jvp_ref, rtol=1e-12, atol=1e-12)
    assert grads_fw.keys() == grads_ref.keys()
    for name in grads_ref:
        assert grads_fw[name].norm().item() > 0.0
        torch.testing.assert_close(grads_fw[name], grads_ref[name], rtol=1e-11, atol=1e-12)

    # 3. Independent central finite-difference check on Hopfield memory & attention weights
    eps = 1e-6
    for param_name, coord in (("mem", (1, 3)), ("qkv.weight", (2, 4)), ("mha.in_proj_weight", (0, 2))):
        param = dict(critic.named_parameters())[param_name]
        orig_val = float(param.data[coord].item())
        with torch.no_grad():
            param.data[coord] = orig_val + eps
            _, _, loss_plus = eval_fwad_loss()
            param.data[coord] = orig_val - eps
            _, _, loss_minus = eval_fwad_loss()
            param.data[coord] = orig_val
        fd_grad = float((loss_plus - loss_minus).item()) / (2.0 * eps)
        analytic_grad = float(grads_fw[param_name][coord].item())
        assert analytic_grad == pytest.approx(fd_grad, rel=1e-5, abs=1e-6)

    # 4. Non-softmax MLP critic: bitwise identical to raw fwAD
    torch.manual_seed(99)
    mlp = PlainMLPCritic(in_dim=8).double()
    mlp.zero_grad()
    with fwAD.dual_level():
        yd_plain = mlp(fwAD.make_dual(x, u))
        y_plain, jvp_plain = fwAD.unpack_dual(yd_plain)
    (jvp_plain.square().mean() + 0.03 * y_plain.mean().square()).backward()
    grads_plain = [p.grad.clone() for p in mlp.parameters()]

    mlp.zero_grad()
    with critic_penalty_context():
        with fwAD.dual_level():
            yd_mode = mlp(fwAD.make_dual(x, u))
            y_mode, jvp_mode = fwAD.unpack_dual(yd_mode)
    (jvp_mode.square().mean() + 0.03 * y_mode.mean().square()).backward()
    grads_mode = [p.grad.clone() for p in mlp.parameters()]

    torch.testing.assert_close(y_mode, y_plain, rtol=0, atol=0)
    torch.testing.assert_close(jvp_mode, jvp_plain, rtol=0, atol=0)
    for g_m, g_p in zip(grads_mode, grads_plain):
        torch.testing.assert_close(g_m, g_p, rtol=0, atol=0)

    # 5. Low-precision bfloat16 forward-AD softmax + backward succeeds with finite gradients
    bf_lin = nn.Linear(8, 8, dtype=torch.bfloat16)
    bf_head = nn.Linear(8, 1, dtype=torch.bfloat16)
    x_bf = x.to(dtype=torch.bfloat16)
    u_bf = u.to(dtype=torch.bfloat16)
    with DualSafeSoftmaxMode():
        with fwAD.dual_level():
            yd_bf = bf_head(torch.softmax(bf_lin(fwAD.make_dual(x_bf, u_bf)), dim=-1))
            _, jvp_bf = fwAD.unpack_dual(yd_bf)
    jvp_bf.float().square().mean().backward()
    assert bf_lin.weight.grad is not None and torch.isfinite(bf_lin.weight.grad).all()
    assert bf_head.weight.grad is not None and torch.isfinite(bf_head.weight.grad).all()

    # 6. invoke_critic_penalty never swallows real exceptions from penalty_fn
    def failing_penalty(_critic, _real, _fake):
        raise ValueError("sentinel failure")

    with pytest.raises(ValueError, match="sentinel failure"):
        invoke_critic_penalty(failing_penalty, mlp, x, x)
