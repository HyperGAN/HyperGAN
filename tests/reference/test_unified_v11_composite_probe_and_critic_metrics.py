"""End-to-end verification of Unified-v11 composite generator probing and per-critic metrics."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import hypergan as hg
from hypergan.checkpoints import read_checkpoint
from hypergan.config import resolve_config
from hypergan.cpu_workers import launch_cpu_workers
from hypergan.distributed_training import ReplicatedCPUTrainer
from hypergan.model_description import describe_config
from hypergan.training import ReferenceTrainer, resume, train


# ---------------------------------------------------------------------------
# Importable synthetic composite fixture components
# ---------------------------------------------------------------------------

class SharedTrunk(nn.Module):
    """Shared upstream trainable trunk with optional BatchNorm/Dropout/RNG usage."""

    def __init__(self, in_dim: int = 4, hidden_dim: int = 6, use_bn: bool = False, touch_rng: bool = False):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim) if use_bn else None
        self.drop = nn.Dropout(0.25) if use_bn else None
        self.touch_rng = touch_rng

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.touch_rng:
            _ = torch.randn(1)
            _ = random.random()
            _ = float(np.random.rand())
        h = self.proj(z)
        if self.bn is not None:
            h = self.bn(h)
        h = torch.tanh(h)
        if self.drop is not None:
            h = self.drop(h)
        return h


class TrainableBranch(nn.Module):
    """Trainable branch taking upstream features and optional conditioning tensor."""

    def __init__(self, in_dim: int = 6, cond_dim: int = 0, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, out_dim),
        )

    def forward(self, h: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if condition is not None:
            h = torch.cat([h, condition], dim=-1)
        return self.net(h)


class ParameterFreeCombiner(nn.Module):
    """Parameter-free output combiner summing two upstream branches."""

    def __init__(self, scale_a: float = 1.0, scale_b: float = 1.0):
        super().__init__()
        self.scale_a = float(scale_a)
        self.scale_b = float(scale_b)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.scale_a * a + self.scale_b * b


class MarginBiasedCritic(nn.Module):
    """Trainable critic initialized to produce a strong, predictable score slope/margin.

    ``sign = +1.0`` scores real (>0) higher than fake (positive margin);
    ``sign = -1.0`` scores real (<0) lower than fake (negative margin),
    so a last-term-only reporting bug is immediately detectable.
    """

    def __init__(self, in_dim: int = 2, sign: float = 1.0, offset: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
        with torch.no_grad():
            self.linear.weight.fill_(float(sign) * 2.5)
            self.linear.bias.fill_(float(offset))

    def forward(self, candidate: torch.Tensor) -> torch.Tensor:
        return self.linear(candidate)


class SyntheticConditionedData:
    """Stateless paired data generator with positive-mean real data for clear critic margins."""

    resume_stateless = True

    def __init__(self, out_dim: int = 2, cond_dim: int = 2, real_shift: float = 2.0):
        self.out_dim = out_dim
        self.cond_dim = cond_dim
        self.real_shift = float(real_shift)

    def __call__(self, batch_size: int, *, generator: torch.Generator) -> dict[str, torch.Tensor]:
        cond = torch.randn(batch_size, self.cond_dim, generator=generator)
        real = self.real_shift + 0.1 * torch.randn(batch_size, self.out_dim, generator=generator)
        return {"real": real, "condition": cond}


def _factory(cls: type) -> str:
    return f"{__name__}:{cls.__name__}"


def _composite_multi_critic_config(
    *,
    steps: int = 3,
    batch_size: int = 8,
    use_bn: bool = False,
    touch_rng: bool = False,
    include_aux_only: bool = False,
    include_shared_critic_term: bool = False,
    every_steps: int = 1,
    sampling_generated: str | None = None,
    lazy_k: int = 1,
    initialization: str = "constructor",
    extra_penalty: bool = True,
) -> dict:
    components: dict = {
        "shared_trunk": {
            "factory": _factory(SharedTrunk),
            "args": {"in_dim": 4, "hidden_dim": 6, "use_bn": use_bn, "touch_rng": touch_rng},
            "inputs": {"z": "latent"},
        },
        "branch_a": {
            "factory": _factory(TrainableBranch),
            "args": {"in_dim": 6, "cond_dim": 0, "out_dim": 2},
            "inputs": {"h": "components.shared_trunk"},
        },
        "branch_b": {
            "factory": _factory(TrainableBranch),
            "args": {"in_dim": 6, "cond_dim": 2, "out_dim": 2},
            "inputs": {"h": "components.shared_trunk", "condition": "batch.condition"},
        },
        "generator": {
            "factory": _factory(ParameterFreeCombiner),
            "args": {"scale_a": 1.0, "scale_b": 1.0},
            "inputs": {"a": "components.branch_a", "b": "components.branch_b"},
        },
        "discriminator": {
            "factory": _factory(MarginBiasedCritic),
            "args": {"in_dim": 2, "sign": 1.0, "offset": 0.5},
            "inputs": {"candidate": "candidate"},
        },
        "aux_critic": {
            "factory": _factory(MarginBiasedCritic),
            "args": {"in_dim": 2, "sign": -1.0, "offset": -0.5},
            "inputs": {"candidate": "candidate"},
        },
    }
    objectives = []
    if include_aux_only:
        components["aux_only"] = {
            "factory": _factory(TrainableBranch),
            "args": {"in_dim": 2, "cond_dim": 0, "out_dim": 2},
            "inputs": {"h": "batch.condition"},
        }
        objectives.append({
            "id": "aux_recon",
            "factory": "mse",
            "weight": 0.1,
            "inputs": {"input": "components.aux_only", "target": "batch.real"},
            "detach": ["target"],
        })
    if sampling_generated is not None:
        components["output_combiner"] = {
            "factory": _factory(ParameterFreeCombiner),
            "args": {"scale_a": 0.5, "scale_b": 0.5},
            "inputs": {"a": "components.branch_a", "b": "components.branch_b"},
        }
    adversarial_terms = [
        {
            "id": "aux_adv",
            "component": "aux_critic",
            "weight": 0.35,
            "critic_weight": 0.35,
            "generator_weight": 0.35,
            "real": "batch.real",
            "fake": "generated" if sampling_generated is None else sampling_generated,
            "inputs": {"candidate": "candidate"},
            "penalty": extra_penalty,
            "penalty_coeff": 0.25 if extra_penalty else 0.0,
        }
    ]
    if include_shared_critic_term:
        adversarial_terms.append({
            "id": "shared_disc_branch_a",
            "component": "discriminator",
            "weight": 0.2,
            "critic_weight": 0.2,
            "generator_weight": 0.2,
            "real": "batch.real",
            "fake": "components.branch_a",
            "inputs": {"candidate": "candidate"},
            "penalty": False,
        })
    cfg = {
        "defaults": "particlegan:unified_v11",
        "data": {"factory": _factory(SyntheticConditionedData), "args": {"out_dim": 2, "cond_dim": 2, "real_shift": 2.0}},
        "prior": {"kind": "mog", "fixed_sigma": 0.05, "args": {"num_particles": 16, "z_dim": 4}},
        "components": components,
        "adversarial": {
            "id": "primary_adv",
            "weight": 1.75,
            "critic_weight": 1.75,
            "generator_weight": 1.75,
            **({"fake": sampling_generated} if sampling_generated is not None else {}),
        },
        "adversarial_terms": adversarial_terms,
        "gradient_penalty": {"coeff": 0.5, "lazy_k": lazy_k},
        "objectives": objectives,
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "seed": 17,
            "device": "cpu",
            "initialization": initialization,
        },
        "sampling": {"count": 4, "seed": 23, **({"generated": sampling_generated} if sampling_generated is not None else {})},
        "metrics": {"every_steps": every_steps},
    }
    return cfg


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


# ---------------------------------------------------------------------------
# 1. Complete generator probe replay, parameter ownership, RNG/mode/BN safety
# ---------------------------------------------------------------------------

def test_composite_generator_probe_replay_ownership_and_state_preservation():
    """Full-graph probe replays from latent + conditioning to output, sees perturbations in
    every upstream branch, deduplicates shared components, keeps critic/prior/aux roles intact,
    and preserves RNG streams, module modes, and BatchNorm running statistics."""
    config = resolve_config(_composite_multi_critic_config(
        use_bn=True, touch_rng=True, include_aux_only=True, steps=2))
    trainer = ReferenceTrainer(config)

    # Output combiner itself is parameter-free:
    assert list(trainer.graph.models["generator"].parameters()) == []

    # Probe parameter set includes shared_trunk (once), branch_a, branch_b — and excludes critics, prior, aux_only:
    expected_ids = {
        id(p)
        for name in ("shared_trunk", "branch_a", "branch_b")
        for p in trainer.graph.models[name].parameters()
        if p.requires_grad
    }
    assert trainer.opt_g._gen_param_ids == expected_ids
    assert len(trainer._generator_probe_params) == len(expected_ids)

    critic_ids = {
        id(p)
        for name in ("discriminator", "aux_critic")
        for p in trainer.graph.models[name].parameters()
    }
    prior_ids = {id(p) for p in trainer.prior.parameters()}
    aux_only_ids = {id(p) for p in trainer.graph.models["aux_only"].parameters()}
    assert trainer.opt_g._gen_param_ids.isdisjoint(critic_ids)
    assert trainer.opt_g._gen_param_ids.isdisjoint(prior_ids)
    assert trainer.opt_g._gen_param_ids.isdisjoint(aux_only_ids)

    # Optimizer role checks: critics in critic_optimizers, prior in prior role, generator params in generator role:
    assert set(trainer.critic_optimizers) == {"discriminator", "aux_critic"}
    roles = [g["role"] for g in trainer.opt_g.param_groups]
    assert roles == ["generator", "prior"]
    gen_group_ids = {id(p) for p in trainer.opt_g.param_groups[0]["params"]}
    assert expected_ids.issubset(gen_group_ids)
    assert aux_only_ids.issubset(gen_group_ids)
    assert gen_group_ids.isdisjoint(critic_ids)
    assert gen_group_ids.isdisjoint(prior_ids)

    # Checkpoint graph state_dict / named_modules does not expose the probe adapter:
    assert "_generator_probe" not in dict(trainer.graph.named_modules())
    assert trainer.ema_graph._probe_hook is None

    # Capture probe during training draw:
    batch = trainer.batch()
    z, _ids = trainer.prior.sample(config["training"]["batch_size"], generator=trainer.streams["prior"])
    trainer.graph.train()
    trainer.graph.generate(z, batch, prior=trainer.prior)
    rec = trainer.opt_d.record
    assert rec._g_probe_cache is not None

    bn = trainer.graph.models["shared_trunk"].bn
    bn_mean = bn.running_mean.detach().clone()
    bn_var = bn.running_var.detach().clone()
    bn_tracked = bn.num_batches_tracked.detach().clone()
    streams_before = {name: s.get_state() for name, s in trainer.streams.items()}
    torch_rng = torch.get_rng_state()
    py_rng = random.getstate()
    np_rng = np.random.get_state()

    g_mod, p_args, p_kwargs, y_pre = rec._g_probe_cache
    try:
        from particlegan._unified_v11.critic import _eval_v11_probe_forward as eval_probe
    except ImportError:
        from particlegan._unified_v4.optimizer import _eval_probe_forward as eval_probe

    # Identical inputs + unchanged weights reproduces y_pre within numerical tolerance:
    y_same = eval_probe(g_mod, p_args, p_kwargs, rec)
    assert y_same is not None
    torch.testing.assert_close(y_same, y_pre, atol=1e-6, rtol=1e-6)

    # Replay preserved RNG streams, module modes, and BN running stats:
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert random.getstate() == py_rng
    assert np.array_equal(np.random.get_state()[1], np_rng[1])
    for name, stream in trainer.streams.items():
        assert torch.equal(stream.get_state(), streams_before[name])
    assert trainer.graph.models["shared_trunk"].training is True
    assert bn.training is True
    assert torch.equal(bn.running_mean, bn_mean)
    assert torch.equal(bn.running_var, bn_var)
    assert torch.equal(bn.num_batches_tracked, bn_tracked)

    # Perturbing branch_a changes replayed output:
    with torch.no_grad():
        trainer.graph.models["branch_a"].net[2].weight.add_(0.5)
    y_pert_a = eval_probe(g_mod, p_args, p_kwargs, rec)
    assert y_pert_a is not None and not torch.allclose(y_pert_a, y_pre, atol=1e-4)
    with torch.no_grad():
        trainer.graph.models["branch_a"].net[2].weight.sub_(0.5)

    # Perturbing branch_b changes replayed output:
    with torch.no_grad():
        trainer.graph.models["branch_b"].net[2].weight.add_(0.5)
    y_pert_b = eval_probe(g_mod, p_args, p_kwargs, rec)
    assert y_pert_b is not None and not torch.allclose(y_pert_b, y_pre, atol=1e-4)
    with torch.no_grad():
        trainer.graph.models["branch_b"].net[2].weight.sub_(0.5)

    # Perturbing shared_trunk changes replayed output:
    with torch.no_grad():
        trainer.graph.models["shared_trunk"].proj.weight.add_(0.5)
    y_pert_shared = eval_probe(g_mod, p_args, p_kwargs, rec)
    assert y_pert_shared is not None and not torch.allclose(y_pert_shared, y_pre, atol=1e-4)
    with torch.no_grad():
        trainer.graph.models["shared_trunk"].proj.weight.sub_(0.5)


def test_configured_sampling_generated_binding_replays_full_subgraph():
    """When ``sampling.generated`` selects an explicit combiner component, the probe replays to it."""
    config = resolve_config(_composite_multi_critic_config(
        sampling_generated="components.output_combiner", steps=2))
    trainer = ReferenceTrainer(config)
    assert trainer._generator_probe._output_path == "components.output_combiner"
    replays = {"count": 0}
    orig_fwd = trainer._generator_probe.forward

    def counted(*args, **kwargs):
        replays["count"] += 1
        return orig_fwd(*args, **kwargs)

    trainer._generator_probe.forward = counted
    row, _ = trainer.update()
    assert replays["count"] == 1
    assert trainer.opt_d.record._g_probe_cache is None
    if hasattr(trainer.opt_g, "last_probe_consumed"):
        assert trainer.opt_g.last_probe_consumed is True
        assert trainer.opt_g.probe_consumptions == 1
    assert row["loss/critics/primary_adv/d_total"] + row["loss/critics/aux_adv/d_total"] == pytest.approx(
        row["d_loss"], rel=1e-5, abs=1e-5
    )


# ---------------------------------------------------------------------------
# 2. End-to-end training entry point: direct probe consumption & per-critic metrics
# ---------------------------------------------------------------------------

def test_train_entrypoint_composite_probe_consumption_and_exact_per_critic_metrics(tmp_path, monkeypatch):
    """Exercise normal ``hypergan.training.train`` end to end:
    - Direct Unified-v11 probe capture & consumption (including gain == 1.0 steps).
    - Exact agreement of all per-critic metrics with training-time tensors and zero extra critic calls.
    - Distinct margins across critics (catching any last-term-only bug) and shared-critic term disambiguation.
    - Metrics API, immutable catalog labels, and model description exposure."""
    config = _composite_multi_critic_config(
        steps=4,
        batch_size=8,
        use_bn=False,
        include_shared_critic_term=True,
        every_steps=2,
        lazy_k=2,
    )
    run_dir = tmp_path / "composite_run"

    step_probe_events = []
    step_training_tensors = []
    critic_forward_counts = []

    orig_init = ReferenceTrainer._initialize

    def instrumented_init(self, cfg):
        orig_init(self, cfg)
        probe_mod = self._generator_probe
        orig_probe_forward = probe_mod.forward
        replay_counter = {"count": 0}

        def counted_probe_forward(*args, **kwargs):
            replay_counter["count"] += 1
            return orig_probe_forward(*args, **kwargs)

        probe_mod.forward = counted_probe_forward

        # Track critic forward call counts per step to guarantee zero extra critic evaluations:
        call_counts = {name: 0 for name in ("discriminator", "aux_critic")}
        for cname in call_counts:
            mod = self.graph.models[cname]
            orig_fwd = mod.forward

            def make_wrapped(name_c, fn):
                def wrapped(*args, **kwargs):
                    call_counts[name_c] += 1
                    return fn(*args, **kwargs)
                return wrapped

            mod.forward = make_wrapped(cname, orig_fwd)

        # Record exact training-time per-term tensors from d_loss & penalty during critic phase:
        captured_step = {}
        critic_term_cursor = {"idx": 0}
        from hypergan import objective_program as op_mod
        orig_bound_scores = op_mod._bound_scores

        def spy_bound_scores(term, context, graph, phase_name, phase, *, first, noise):
            result = orig_bound_scores(term, context, graph, phase_name, phase, first=first, noise=noise)
            if phase_name == "critic":
                _, _, r_sc, f_sc = result
                captured_step.setdefault(term.id, {})["real_mean"] = float(r_sc.detach().double().mean().item())
                captured_step[term.id]["fake_mean"] = float(f_sc.detach().double().mean().item())
                captured_step[term.id]["critic_weight"] = float(term.critic_weight)
                critic_term_cursor["current"] = term
            return result

        monkeypatch.setattr(op_mod, "_bound_scores", spy_bound_scores)

        wrapped_gans = set()
        for term in self.program.adversarial_terms:
            if id(term.gan) not in wrapped_gans:
                wrapped_gans.add(id(term.gan))
                orig_d_loss = term.gan.d_loss

                def make_d_loss_spy(fn):
                    def spy_d_loss(r_sc, f_sc, *args, **kwargs):
                        val = fn(r_sc, f_sc, *args, **kwargs)
                        active_term = critic_term_cursor["current"]
                        captured_step.setdefault(active_term.id, {})["d_adv_weighted"] = float(
                            (active_term.critic_weight * val).detach().item()
                        )
                        return val
                    return spy_d_loss

                term.gan.d_loss = make_d_loss_spy(orig_d_loss)

            if term.penalty_fn is not None:
                orig_pen = term.penalty_fn

                def make_pen_spy(tid, fn):
                    def spy_pen(*args, **kwargs):
                        val = fn(*args, **kwargs)
                        captured_step.setdefault(tid, {})["d_pen"] = float(val.detach().item())
                        return val
                    return spy_pen

                object.__setattr__(term, "penalty_fn", make_pen_spy(term.id, orig_pen))

        orig_opt_step = self.opt_g.step

        def spy_opt_g_step(*args, **kwargs):
            rec = self.opt_d.record
            had_cache = rec._g_probe_cache is not None
            replays_before = replay_counter["count"]
            out = orig_opt_step(*args, **kwargs)
            step_probe_events.append({
                "had_cache": had_cache,
                "cache_cleared": rec._g_probe_cache is None,
                "last_probe_consumed": getattr(self.opt_g, "last_probe_consumed", None),
                "probe_consumptions": getattr(self.opt_g, "probe_consumptions", None),
                "replay_delta": replay_counter["count"] - replays_before,
                "last_step_gain": self.opt_g.last_step_gain,
            })
            return out

        self.opt_g.step = spy_opt_g_step

        orig_update = self._update

        def spy_update(*args, **kwargs):
            captured_step.clear()
            for k in call_counts:
                call_counts[k] = 0
            row, batch_out = orig_update(*args, **kwargs)
            step_training_tensors.append((row["step"], copy.deepcopy(captured_step)))
            critic_forward_counts.append((row["step"], dict(call_counts)))
            return row, batch_out

        self._update = spy_update

    monkeypatch.setattr(ReferenceTrainer, "_initialize", instrumented_init)

    train(config, run_dir, checkpoint_every=4)

    # 1. Direct assertion of Unified-v11 probe capture & consumption on every step:
    assert len(step_probe_events) == 4
    for idx, ev in enumerate(step_probe_events, start=1):
        assert ev["had_cache"] is True
        assert ev["cache_cleared"] is True
        assert ev["replay_delta"] == 1
        if ev["last_probe_consumed"] is not None:
            assert ev["last_probe_consumed"] is True
            assert ev["probe_consumptions"] == idx
    # A valid probe can produce a gain of exactly 1.0; consumption was still directly verified:
    assert all(ev["last_step_gain"] > 0.0 for ev in step_probe_events)
    assert any(ev["last_step_gain"] == 1.0 for ev in step_probe_events)

    # 2. Verify zero extra critic evaluations:
    # For discriminator (used by primary_adv + shared_disc_branch_a):
    #   D-phase: 2 terms * 2 scores (real+fake) = 4 forwards (+ 1 fwAD penalty forward on lazy_k steps via live endpoint reuse)
    #   G-phase: 2 terms * 2 scores (fake+real) = 4 forwards
    # For aux_critic (used by aux_adv):
    #   D-phase: 2 scores (+ 1 fwAD penalty forward on lazy_k steps)
    #   G-phase: 2 scores
    for step_num, counts in critic_forward_counts:
        pen_calls = 1 if step_num % 2 == 0 else 0
        assert counts["discriminator"] == 8 + pen_calls
        assert counts["aux_critic"] == 4 + pen_calls

    # 3. Verify metrics API, cadence (every_steps=2 -> steps 2, 4), catalog labels, and exact tensor match:
    series = hg.metrics(run_dir)
    catalog = hg.catalog(run_dir)["metrics"]
    term_ids = ("primary_adv", "aux_adv", "shared_disc_branch_a")

    # Preserve existing aggregate loss series:
    for agg in ("loss/d_total", "loss/d_adversarial", "loss/d_adversarial_raw", "loss/gradient_penalty", "loss/g_total"):
        assert agg in series
        assert [s for s, _ in series[agg]] == [2, 4]

    # Margin charts identify their term:
    assert "primary_adv" in catalog["critic/margin"]["label"]
    assert catalog["critic/margin"]["term_id"] == "primary_adv"
    for tid in term_ids:
        assert tid in catalog[f"critic/{tid}/margin"]["label"]
        assert catalog[f"critic/{tid}/margin"]["term_id"] == tid
        for suffix in ("d_adversarial", "gradient_penalty", "d_total"):
            assert f"loss/critics/{tid}/{suffix}" in series
            assert [s for s, _ in series[f"loss/critics/{tid}/{suffix}"]] == [2, 4]
        for suffix in ("real_mean", "fake_mean", "margin"):
            assert f"critic/{tid}/{suffix}" in series
            assert [s for s, _ in series[f"critic/{tid}/{suffix}"]] == [2, 4]

    # Shared-critic terms remain distinct:
    assert catalog["loss/critics/primary_adv/d_adversarial"]["critic"] == "discriminator"
    assert catalog["loss/critics/shared_disc_branch_a/d_adversarial"]["critic"] == "discriminator"
    assert catalog["loss/critics/primary_adv/d_adversarial"]["term_id"] != catalog["loss/critics/shared_disc_branch_a/d_adversarial"]["term_id"]

    # Check exact agreement with training-time tensors & sum identity on every published step:
    by_step_tensors = dict(step_training_tensors)
    for idx, step_num in enumerate((2, 4)):
        rec_step = by_step_tensors[step_num]
        term_total_sum = 0.0
        term_adv_sum = 0.0
        term_pen_sum = 0.0
        for tid in term_ids:
            expected_adv = rec_step[tid]["d_adv_weighted"]
            expected_pen = rec_step[tid].get("d_pen", 0.0)
            expected_tot = expected_adv + expected_pen
            expected_real = rec_step[tid]["real_mean"]
            expected_fake = rec_step[tid]["fake_mean"]
            expected_margin = expected_real - expected_fake

            pub_adv = series[f"loss/critics/{tid}/d_adversarial"][idx][1]
            pub_pen = series[f"loss/critics/{tid}/gradient_penalty"][idx][1]
            pub_tot = series[f"loss/critics/{tid}/d_total"][idx][1]
            pub_real = series[f"critic/{tid}/real_mean"][idx][1]
            pub_fake = series[f"critic/{tid}/fake_mean"][idx][1]
            pub_margin = series[f"critic/{tid}/margin"][idx][1]

            assert pub_adv == pytest.approx(expected_adv, rel=1e-5, abs=1e-5)
            assert pub_pen == pytest.approx(expected_pen, rel=1e-5, abs=1e-5)
            assert pub_tot == pytest.approx(expected_tot, rel=1e-5, abs=1e-5)
            assert pub_real == pytest.approx(expected_real, rel=1e-5, abs=1e-5)
            assert pub_fake == pytest.approx(expected_fake, rel=1e-5, abs=1e-5)
            assert pub_margin == pytest.approx(expected_margin, rel=1e-5, abs=1e-5)

            term_adv_sum += pub_adv
            term_pen_sum += pub_pen
            term_total_sum += pub_tot

        pub_d_total = series["loss/d_total"][idx][1]
        pub_d_adv = series["loss/d_adversarial"][idx][1]
        pub_d_pen = series["loss/gradient_penalty"][idx][1]
        assert term_total_sum == pytest.approx(pub_d_total, rel=1e-5, abs=1e-5)
        assert term_adv_sum == pytest.approx(pub_d_adv, rel=1e-5, abs=1e-5)
        assert term_pen_sum == pytest.approx(pub_d_pen, rel=1e-5, abs=1e-5)

        # Deliberately different margins between primary_adv (>0) and aux_adv (<0):
        m_primary = series["critic/primary_adv/margin"][idx][1]
        m_aux = series["critic/aux_adv/margin"][idx][1]
        assert m_primary > 0.5
        assert m_aux < -0.5
        assert abs(m_primary - m_aux) > 1.0
        # Legacy top-level critic/margin must match primary_adv (NOT last term!):
        assert series["critic/margin"][idx][1] == pytest.approx(m_primary, rel=1e-5, abs=1e-5)

    # Model description exposes clickable per-term discriminator series:
    desc = describe_config(resolve_config(config))
    d_terms = {t["id"]: t for t in desc["losses"]["discriminator"]}
    for tid in term_ids:
        assert d_terms[tid]["metric"] == f"loss/critics/{tid}/d_adversarial"
    assert d_terms["primary_adv:penalty"]["metric"] == "loss/critics/primary_adv/gradient_penalty"
    assert d_terms["aux_adv:penalty"]["metric"] == "loss/critics/aux_adv/gradient_penalty"


# ---------------------------------------------------------------------------
# 3. Single-module generator & checkpoint/resume contract
# ---------------------------------------------------------------------------

def test_single_module_generator_unified_v11_probe_preserved(tmp_path):
    """Ordinary single-module generator still captures/consumes its probe and publishes metrics."""
    config = {
        "defaults": "particlegan:unified_v11",
        "data": {"factory": _factory(SyntheticConditionedData), "args": {"out_dim": 2, "cond_dim": 2, "real_shift": 2.0}},
        "prior": {"kind": "mog", "fixed_sigma": 0.05, "args": {"num_particles": 16, "z_dim": 4}},
        "components": {
            "generator": {
                "factory": "hndl",
                "args": {"source": "linear(8)\nleaky_relu(0.2)\nlinear()", "input_shape": ["B", 4], "output_shape": ["B", 2]},
                "inputs": {"x": "latent"},
            },
            "discriminator": {
                "factory": _factory(MarginBiasedCritic),
                "args": {"in_dim": 2, "sign": 1.0, "offset": 0.0},
                "inputs": {"candidate": "candidate"},
            },
        },
        "training": {"steps": 2, "batch_size": 8, "seed": 5, "device": "cpu"},
        "sampling": {"count": 4, "seed": 9},
        "metrics": {"every_steps": 1},
    }
    trainer = ReferenceTrainer(resolve_config(config))
    assert trainer.opt_g._gen_param_ids == {id(p) for p in trainer.graph.models["generator"].parameters() if p.requires_grad}

    replays = {"count": 0}
    orig_fwd = trainer._generator_probe.forward

    def counted(*args, **kwargs):
        replays["count"] += 1
        return orig_fwd(*args, **kwargs)

    trainer._generator_probe.forward = counted
    row, _ = trainer.update()
    assert replays["count"] == 1
    assert trainer.opt_d.record._g_probe_cache is None
    if hasattr(trainer.opt_g, "last_probe_consumed"):
        assert trainer.opt_g.last_probe_consumed is True
        assert trainer.opt_g.probe_consumptions == 1
    assert row["loss/critics/adversarial/d_total"] == pytest.approx(row["d_loss"], rel=1e-5, abs=1e-5)
    assert row["critic/adversarial/margin"] == pytest.approx(row["critic/margin"], rel=1e-5, abs=1e-5)

    run_dir = tmp_path / "single_run"
    train(config, run_dir, checkpoint_every=2)
    series = hg.metrics(run_dir)
    assert len(series["loss/critics/adversarial/d_total"]) == 2
    assert len(series["critic/adversarial/margin"]) == 2


def test_composite_unified_v11_checkpoint_resume_continuation(tmp_path):
    """Checkpoint & resume on composite multi-critic Unified-v11 preserves bit-exact state and metrics."""
    config = _composite_multi_critic_config(steps=4, batch_size=8, use_bn=True, every_steps=1)
    full_dir = tmp_path / "full"
    split_dir = tmp_path / "split"

    train(config, full_dir, checkpoint_every=4)
    train(config, split_dir, checkpoint_every=2, stop_after_steps=2)
    resume(split_dir, checkpoint_every=2)

    _, _, full_state = read_checkpoint(full_dir)
    _, _, split_state = read_checkpoint(split_dir)
    _assert_nested_equal(full_state["graph"], split_state["graph"])
    _assert_nested_equal(full_state["prior"], split_state["prior"])
    _assert_nested_equal(full_state["ema_graph"], split_state["ema_graph"])
    _assert_nested_equal(full_state["optimizers"], split_state["optimizers"])

    full_metrics = hg.metrics(full_dir)
    split_metrics = hg.metrics(split_dir)
    for key in (
        "loss/d_total",
        "loss/critics/primary_adv/d_total",
        "loss/critics/aux_adv/d_total",
        "critic/primary_adv/margin",
        "critic/aux_adv/margin",
    ):
        assert full_metrics[key] == split_metrics[key]


# ---------------------------------------------------------------------------
# 4. Distributed CPU/Gloo regression (composite probe + per-critic metrics)
# ---------------------------------------------------------------------------

def _distributed_composite_worker(rank, world_size, config_dict, full_batches, shared_dir, accumulation_steps):
    config = resolve_config(config_dict)
    trainer = ReplicatedCPUTrainer(config, world_size=world_size, accumulation_steps=accumulation_steps)
    rows = []
    probe_consumed = []
    replays = {"count": 0}
    if getattr(trainer, "_generator_probe", None) is not None:
        orig_fwd = trainer._generator_probe.forward

        def counted(*args, **kwargs):
            replays["count"] += 1
            return orig_fwd(*args, **kwargs)

        trainer._generator_probe.forward = counted
    for full_batch in full_batches:
        local_batch = {
            "real": full_batch["real"].chunk(world_size)[rank],
            "condition": full_batch["condition"].chunk(world_size)[rank],
        }
        local_ids = full_batch["_ids"].chunk(world_size)[rank]
        prior_arg = (trainer.prior.z[local_ids], local_ids)
        before = replays["count"]
        row, _ = trainer.update(local_batch, prior_arg)
        rows.append(row)
        probe_consumed.append([
            getattr(trainer.opt_g, "last_probe_consumed", None),
            getattr(trainer.opt_g, "probe_consumptions", None),
            replays["count"] - before,
        ])
    trainer._assert_replicas("composite multi-critic distributed")
    out = {
        "rows": rows,
        "probe_consumed": probe_consumed,
        "digest": trainer._replica_digest(),
    }
    (Path(shared_dir) / f"rank-{rank}.json").write_text(json.dumps(out), encoding="utf-8")


@pytest.mark.parametrize("accumulation_steps", [1, 2])
def test_distributed_gloo_composite_probe_and_per_critic_metric_reduction(tmp_path, accumulation_steps):
    """2-rank CPU/Gloo replicated run matches ReferenceTrainer per-critic definitions,
    agrees bitwise across ranks, sums per-term totals to d_loss, keeps margins distinct,
    and consumes the composite Unified-v11 generator probe on both ranks."""
    config_dict = _composite_multi_critic_config(
        steps=2,
        batch_size=8,
        use_bn=False,
        include_shared_critic_term=True,
        every_steps=1,
        extra_penalty=(accumulation_steps == 1),
    )
    if accumulation_steps > 1:
        # Accumulation requires a non-coupled loss (legacy hypergan recipe is standard for acc>1);
        # exercise accumulation reduction on multi-critic terms with unequal weights & distinct margins.
        config_dict.pop("defaults")
        config_dict["training"]["initialization"] = "constructor"
        config_dict["optimizer"] = {"lr": 1e-3}

    gen = torch.Generator().manual_seed(99)
    full_batches = [
        {
            "real": 2.0 + 0.1 * torch.randn(8, 2, generator=gen),
            "condition": torch.randn(8, 2, generator=gen),
            "_ids": torch.arange(8, dtype=torch.long),
        }
        for _ in range(2)
    ]

    ref = ReferenceTrainer(resolve_config(copy.deepcopy(config_dict)))
    ref_rows = []
    for fb in full_batches:
        ids = fb["_ids"]
        r_row, _ = ref.update({"real": fb["real"], "condition": fb["condition"]}, (ref.prior.z[ids], ids))
        ref_rows.append(r_row)

    launch_cpu_workers(
        _distributed_composite_worker,
        args=(config_dict, full_batches, str(tmp_path), accumulation_steps),
        world_size=2,
        timeout=60,
        collective_timeout=20,
    )
    r0 = json.loads((tmp_path / "rank-0.json").read_text(encoding="utf-8"))
    r1 = json.loads((tmp_path / "rank-1.json").read_text(encoding="utf-8"))

    # Bitwise rank agreement:
    assert r0["digest"] == r1["digest"]
    assert r0["rows"] == r1["rows"]

    if accumulation_steps == 1:
        # Rank 0 executes the broadcast probe replay on every step (ranks share exact broadcast result):
        assert [row[2] for row in r0["probe_consumed"]] == [1, 1]
        if r0["probe_consumed"][0][0] is not None:
            assert [row[:2] for row in r0["probe_consumed"]] == [[True, 1], [True, 2]]
            assert [row[:2] for row in r1["probe_consumed"]] == [[True, 1], [True, 2]]

    term_ids = ("primary_adv", "aux_adv", "shared_disc_branch_a")
    for step_idx, (dist_row, ref_row) in enumerate(zip(r0["rows"], ref_rows)):
        # Sum of per-term totals equals published discriminator total:
        term_total_sum = sum(dist_row[f"loss/critics/{tid}/d_total"] for tid in term_ids)
        assert term_total_sum == pytest.approx(dist_row["d_loss"], rel=1e-5, abs=1e-5)

        # Deliberately different margins preserved across distributed reduction:
        m_primary = dist_row["critic/primary_adv/margin"]
        m_aux = dist_row["critic/aux_adv/margin"]
        assert m_primary > 0.5
        assert m_aux < -0.5
        assert dist_row["critic/margin"] == pytest.approx(m_primary, rel=1e-6, abs=1e-6)

        # Step 1 (identical initial weights + identical global batch) matches ReferenceTrainer closely:
        if step_idx == 0:
            for tid in term_ids:
                for key in (
                    f"loss/critics/{tid}/d_adversarial",
                    f"loss/critics/{tid}/d_total",
                    f"critic/{tid}/real_mean",
                    f"critic/{tid}/fake_mean",
                    f"critic/{tid}/margin",
                ):
                    assert dist_row[key] == pytest.approx(ref_row[key], rel=1e-4, abs=1e-4), (
                        f"Mismatch on {key}: dist={dist_row[key]} ref={ref_row[key]}"
                    )


def test_viewer_api_exposes_per_critic_catalog_model_and_bootstrap_series(tmp_path):
    """The local viewer HTTP API serves per-critic catalog definitions, model loss bindings, and reduced series."""
    import time
    from starlette.testclient import TestClient
    from hypergan.event_views import Projector
    from hypergan.web_server import create_app
    from hypergan.web_session import LocalSession

    config = _composite_multi_critic_config(
        steps=2,
        batch_size=8,
        include_shared_critic_term=True,
        every_steps=1,
    )
    run_dir = tmp_path / "viewer_run"
    manifest = train(config, run_dir, checkpoint_every=2)
    run_id = manifest["run_id"]
    with Projector(run_dir) as projector:
        projector.project(limit=10000)

    session = LocalSession(8123, auth="none")
    app = create_app(run_dir, session, poll_seconds=0.01)
    with TestClient(app, base_url=session.origin) as client:
        catalog = client.get(f"/api/v1/runs/{run_id}/metrics/catalog").json()["metrics"]
        for tid in ("primary_adv", "aux_adv", "shared_disc_branch_a"):
            assert f"loss/critics/{tid}/d_total" in catalog
            assert f"critic/{tid}/margin" in catalog
            assert tid in catalog[f"critic/{tid}/margin"]["label"]
        assert "primary_adv" in catalog["critic/margin"]["label"]

        model_doc = client.get(f"/api/v1/runs/{run_id}/model").json()
        d_terms = {t["id"]: t for t in model_doc["losses"]["discriminator"]}
        assert d_terms["primary_adv"]["metric"] == "loss/critics/primary_adv/d_adversarial"
        assert d_terms["aux_adv"]["metric"] == "loss/critics/aux_adv/d_adversarial"
        assert d_terms["shared_disc_branch_a"]["metric"] == "loss/critics/shared_disc_branch_a/d_adversarial"

        views = client.get(f"/api/v1/runs/{run_id}/views").json()
        map_rev = views["map_revision"]
        requested = "loss/d_total,loss/critics/primary_adv/d_total,loss/critics/aux_adv/d_total,critic/primary_adv/margin,critic/aux_adv/margin"
        resp = None
        for _ in range(100):
            resp = client.get(f"/api/v1/runs/{run_id}/views/{map_rev}/bootstrap?series={requested}&bucket_steps=1")
            if resp.status_code == 200:
                break
            time.sleep(0.02)
        assert resp is not None and resp.status_code == 200, resp.text if resp is not None else "no response"
        groups = resp.json()["groups"]
        returned_metrics = {g["key"][0] for g in groups}
        assert {
            "loss/d_total",
            "loss/critics/primary_adv/d_total",
            "loss/critics/aux_adv/d_total",
            "critic/primary_adv/margin",
            "critic/aux_adv/margin",
        } <= returned_metrics
