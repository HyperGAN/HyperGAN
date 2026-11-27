"""Single-GPU CUDA tests for BF16 + channels_last training_precision parity with GANTrainer."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from hypergan.artifacts import LoadedGenerator, sample, save_bundle
from hypergan.config import load_config
from hypergan.objective_program import make_scored_critic, score_candidate
from hypergan.training import ReferenceTrainer, resume, train, update_ema


@pytest.fixture(autouse=True)
def require_bf16_cuda():
    assert torch.cuda.is_available(), "CUDA tests require a CUDA-enabled runtime and local GPU"
    assert torch.cuda.is_bf16_supported(including_emulation=False), "BF16 tests require native BF16 GPU"


class _SyntheticImages:
    resume_stateless = True

    def __init__(self, channels=3, size=16):
        self.channels = channels
        self.size = size

    def __call__(self, batch_size, *, generator):
        return {
            "real": torch.randn(batch_size, self.channels, self.size, self.size, generator=generator)
        }


def _conv_model_config(*, precision=None, use_bn=False, size=16, batch_size=8, steps=4, device="cuda:0"):
    base = size // 4
    g_norm = "batch_norm()" if use_bn else "group_norm(8)"
    d_norm = "batch_norm()" if use_bn else ""
    g_src = (
        f"linear({64 * base * base})\nreshape(64, {base}, {base})\n"
        f"{g_norm}\nsilu()\n"
        f"upsample(2)\nconv(64, kernel_size=3, padding=1)\n{g_norm}\nsilu()\n"
        f"upsample(2)\nconv(3, kernel_size=3, padding=1)\ntanh()"
    )
    d_src = (
        f"conv(64, kernel_size=3, stride=2, padding=1)\n"
        f"{d_norm + chr(10) if d_norm else ''}silu()\n"
        f"conv(64, kernel_size=3, stride=2, padding=1)\n"
        f"{d_norm + chr(10) if d_norm else ''}silu()\n"
        f"flatten()\nlinear(1)"
    )
    train_tbl = {
        "steps": steps,
        "batch_size": batch_size,
        "seed": 11,
        "device": device,
        "initialization": "deterministic_orthogonal",
    }
    if precision is not None:
        train_tbl["precision"] = precision
    return load_config(
        {
            "format": "hypergan-model/1",
            "name": "test/cuda-conv-precision",
            "defaults": "particlegan",
            "data": {"factory": f"{__name__}:_SyntheticImages", "args": {"channels": 3, "size": size}},
            "prior": {"kind": "particles", "z_dim": 16, "num_particles": 64, "initialization_seed": 2},
            "networks": {
                "generator": {
                    "role": "generator",
                    "inputs": {"x": "latent"},
                    "input_shape": ["B", 16],
                    "output_shape": ["B", 3, size, size],
                    "source": g_src,
                },
                "discriminator": {
                    "role": "critic",
                    "inputs": {"x": "candidate"},
                    "input_shape": ["B", 3, size, size],
                    "output_shape": ["B", 1],
                    "source": d_src,
                },
            },
            "losses": [
                {"type": "adversarial", "critic": "discriminator", "weight": 1.0, "penalty": 1.0},
                {"type": "prior"},
            ],
            "train": train_tbl,
        }
    )


def _transgan_like_config(*, precision=None, device="cuda:0"):
    g_src = (
        "h = linear(x, 1024)\n"
        "h = reshape(h, 16, 64)\n"
        "h = pos_embed(h, 16)\n"
        "b = rms_norm(h, eps=1e-8, affine=False)\n"
        "b = attention(b, 4, dropout=0.0, qkv_bias=False, out_bias=True)\n"
        "h = add(h, b)\n"
        "h = linear(h, 48)\n"
        "h = transpose(h, 1, 2)\n"
        "h = reshape(h, 48, 4, 4)\n"
        "h = pixel_shuffle(h, 4)\n"
        "out = tanh(h)"
    )
    d_src = (
        "h = patch_embed(x, 64, 4)\n"
        "b = rms_norm(h, eps=1e-8, affine=False)\n"
        "b = attention(b, 4, dropout=0.0, qkv_bias=False, out_bias=True)\n"
        "h = add(h, b)\n"
        "h = pool_tokens(h, \"first\")\n"
        "out = linear(h, 1)"
    )
    train_tbl = {
        "steps": 3,
        "batch_size": 8,
        "seed": 19,
        "device": device,
    }
    if precision is not None:
        train_tbl["precision"] = precision
    return load_config(
        {
            "format": "hypergan-model/1",
            "name": "test/cuda-transgan-precision",
            "defaults": "particlegan",
            "data": {"factory": f"{__name__}:_SyntheticImages", "args": {"channels": 3, "size": 16}},
            "prior": {"kind": "particles", "z_dim": 16, "num_particles": 64, "initialization_seed": 2},
            "networks": {
                "generator": {
                    "role": "generator",
                    "inputs": {"x": "latent"},
                    "input_shape": ["B", 16],
                    "output_shape": ["B", 3, 16, 16],
                    "source": g_src,
                },
                "discriminator": {
                    "role": "critic",
                    "inputs": {"x": "candidate"},
                    "input_shape": ["B", 3, 16, 16],
                    "output_shape": ["B", 1],
                    "source": d_src,
                },
            },
            "losses": [
                {"type": "adversarial", "critic": "discriminator", "weight": 1.0, "penalty": 1.0},
                {"type": "prior"},
            ],
            "train": train_tbl,
        }
    )


def _conv_4d_weights(module):
    return [
        p
        for m in module.modules()
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))
        for p in (m.weight,)
        if p is not None and p.ndim == 4
    ]


def _formulation_bf16_attr(recipe) -> str:
    formulation = getattr(recipe, "effective_critic_formulation", None)
    if formulation == "axiom":
        return "_axiom_bf16_autocast"
    return "_unified_v4_bf16_autocast"


def test_bf16_channels_last_layout_flags_ema_and_live_endpoint_reuse():
    cfg = _conv_model_config(precision="bf16_channels_last", use_bn=False)
    trainer = ReferenceTrainer(cfg)
    form_attr = _formulation_bf16_attr(trainer.recipe)

    assert trainer.recipe.training_precision == "bf16_channels_last"
    assert trainer.graph._bf16_components == frozenset({"generator", "discriminator"})
    assert trainer.graph._channels_last_components == frozenset({"generator", "discriminator"})
    assert trainer._bf16_generator is True
    assert trainer._bf16_critics == frozenset({"discriminator"})
    assert trainer._in_training_step is False
    assert trainer.graph._training_bf16_active is False

    # Canonical and formulation probe flags set on live G, D, and _generator_probe; cleared on ema_graph
    assert getattr(trainer.graph.models["generator"], "_particlegan_bf16_autocast", False) is True
    assert getattr(trainer.graph.models["discriminator"], "_particlegan_bf16_autocast", False) is True
    assert getattr(trainer._generator_probe, "_particlegan_bf16_autocast", False) is True
    assert getattr(trainer.graph.models["generator"], form_attr, False) is True
    assert getattr(trainer.graph.models["discriminator"], form_attr, False) is True
    assert getattr(trainer._generator_probe, form_attr, False) is True
    assert trainer.ema_graph._bf16_components == frozenset()
    assert trainer.ema_graph._channels_last_components == frozenset()
    assert getattr(trainer.ema_graph.models["generator"], "_particlegan_bf16_autocast", False) is False
    assert getattr(trainer.ema_graph.models["discriminator"], "_particlegan_bf16_autocast", False) is False
    assert getattr(trainer.ema_graph.models["generator"], form_attr, False) is False
    assert getattr(trainer.ema_graph.models["discriminator"], form_attr, False) is False

    # Master weights stay FP32 strided; 4D conv weights in BOTH graph and ema_graph are channels_last
    for graph in (trainer.graph, trainer.ema_graph):
        for p in graph.parameters():
            assert p.dtype == torch.float32
            assert p.layout == torch.strided
        for name in ("generator", "discriminator"):
            weights = _conv_4d_weights(graph.models[name])
            assert weights
            assert all(w.is_contiguous(memory_format=torch.channels_last) for w in weights)

    # Verify live-endpoint reuse stays True under channels_last (critic_source_context)
    trainer.penalty.collect_stats = True
    row, _ = trainer.update()
    assert all(math.isfinite(row[k]) for k in ("d_loss", "g_loss", "gradient_penalty"))
    assert trainer.penalty.last_stats.get("live_endpoints_reused") is True
    assert trainer._in_training_step is False
    assert trainer.graph._training_bf16_active is False

    # Optimizer states remain FP32
    for opt in trainer.optimizers:
        for state in opt.state.values():
            for val in state.values():
                if isinstance(val, torch.Tensor) and val.is_floating_point():
                    assert val.dtype == torch.float32

    # Fused EMA update runs with a single _foreach_lerp_ launch and preserves channels_last FP32
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        update_ema(trainer.ema_graph, trainer.graph, trainer.config["training"]["ema"])
    events = {e.key: e.count for e in prof.key_averages()}
    assert events.get("aten::_foreach_lerp_") == 1
    assert all(w.is_contiguous(memory_format=torch.channels_last) for w in _conv_4d_weights(trainer.ema_graph))


def test_explicit_float32_on_cuda_keeps_nchw_and_disables_bf16():
    cfg = _conv_model_config(precision="float32", use_bn=False)
    trainer = ReferenceTrainer(cfg)
    form_attr = _formulation_bf16_attr(trainer.recipe)

    assert trainer.recipe.training_precision == "float32"
    assert trainer.graph._bf16_components == frozenset()
    assert trainer.graph._channels_last_components == frozenset()
    assert trainer._bf16_generator is False
    assert trainer._bf16_critics == frozenset()
    assert getattr(trainer.graph.models["generator"], form_attr, False) is False
    assert getattr(trainer.graph.models["discriminator"], form_attr, False) is False
    assert getattr(trainer._generator_probe, form_attr, False) is False

    for w in _conv_4d_weights(trainer.graph):
        if w.shape[1] > 1 and (w.shape[2] > 1 or w.shape[3] > 1):
            assert w.is_contiguous(memory_format=torch.contiguous_format)
            assert not w.is_contiguous(memory_format=torch.channels_last)

    row, _ = trainer.update()
    assert all(math.isfinite(row[k]) for k in ("d_loss", "g_loss", "gradient_penalty"))


def test_scored_critic_bf16_flag_is_dynamic_and_honors_toggle():
    cfg = _conv_model_config(precision="bf16_channels_last", use_bn=False)
    trainer = ReferenceTrainer(cfg)
    seen_autocast = []

    def _probe_hook(_mod, _args, _out):
        seen_autocast.append(torch.is_autocast_enabled("cuda"))

    handle = trainer.graph.models["discriminator"].register_forward_hook(_probe_hook)
    try:
        batch = trainer.batch()
        z, _ = trainer.prior.sample(len(batch["real"]), generator=trainer.streams["prior"])
        with trainer._training_step_precision():
            context = trainer.graph.generate(z, batch, prior=trainer.prior)
            term = trainer.program.adversarial_terms[0]
            critic = make_scored_critic(term, context, trainer.graph)
            assert critic.bf16 is True

            out_bf16 = critic(batch["real"])
            assert out_bf16.dtype == torch.float32
            assert seen_autocast[-1] is True

            # Dynamically toggling critic.bf16 = False (as penalty.py does on FP32 fallback) disables autocast
            critic.bf16 = False
            out_fp32 = critic(batch["real"])
            assert out_fp32.dtype == torch.float32
            assert seen_autocast[-1] is False

            critic.bf16 = True
            out_bf16_again = critic(batch["real"])
            assert out_bf16_again.dtype == torch.float32
            assert seen_autocast[-1] is True
    finally:
        handle.remove()


def test_bn_dcgan_and_transgan_like_train_cleanly_in_bf16_with_zero_fallback():
    for cfg, expect_gen_bf16 in (
        (_conv_model_config(precision="bf16_channels_last", use_bn=True), True),
        (_transgan_like_config(precision="bf16_channels_last"), False),
    ):
        trainer = ReferenceTrainer(cfg)
        form_attr = _formulation_bf16_attr(trainer.recipe)
        assert trainer._bf16_generator is expect_gen_bf16
        assert "discriminator" in trainer.graph._bf16_components
        assert getattr(trainer._generator_probe, form_attr, False) is expect_gen_bf16

        # Instrument fwAD evaluator to prove BF16 fwAD succeeds on the first try
        # with zero FP32 exception fallback and zero central-difference fallback.
        reg = trainer.penalty.regularizer
        fwad_method = (
            "_probe_batch"
            if hasattr(reg, "_probe_batch")
            else ("_fwad" if hasattr(reg, "_fwad") else "_eval_image_fwad_sq")
        )
        orig_eval = getattr(reg, fwad_method)
        fwad_invocations = []

        def _spy_fwad(*args, **kwargs):
            calls_in_eval = []
            h = trainer.graph.models["discriminator"].register_forward_hook(
                lambda _m, _a, _o: calls_in_eval.append(torch.is_autocast_enabled("cuda"))
            )
            try:
                res = orig_eval(*args, **kwargs)
            finally:
                h.remove()
            fwad_invocations.append(calls_in_eval)
            return res

        setattr(reg, fwad_method, _spy_fwad)
        for _ in range(3):
            row, _ = trainer.update()
            assert all(math.isfinite(row[k]) for k in ("d_loss", "g_loss", "gradient_penalty"))
        assert fwad_invocations, f"expected {fwad_method} to execute during training"
        assert all(calls == [True] for calls in fwad_invocations), (
            f"unexpected fwAD fallback(s): {fwad_invocations}"
        )
        if hasattr(reg, "_fwad_unsupported"):
            assert not reg._fwad_unsupported, f"fwAD fell back to secant: {reg._fwad_unsupported}"


def test_inference_and_sampling_remain_fp32_under_bf16_training(tmp_path: Path):
    cfg = _conv_model_config(precision="bf16_channels_last", use_bn=False, steps=2)
    trainer = ReferenceTrainer(cfg)
    _, batch = trainer.update()

    # Outside training step, both ema_graph and live graph run strictly in FP32 (no autocast)
    seen_autocast = []
    h_ema = trainer.ema_graph.models["generator"].register_forward_hook(
        lambda _m, _a, _o: seen_autocast.append(("ema", torch.is_autocast_enabled("cuda")))
    )
    h_live = trainer.graph.models["generator"].register_forward_hook(
        lambda _m, _a, _o: seen_autocast.append(("live", torch.is_autocast_enabled("cuda")))
    )
    try:
        z, _ = trainer.ema_prior.sample(4, generator=trainer.streams["prior"])
        with torch.no_grad():
            ema_out = trainer.ema_graph.generate(z, batch, prior=trainer.ema_prior)["generated"]
            live_out = trainer.graph.generate(z, batch, prior=trainer.prior)["generated"]
        assert ema_out.dtype == torch.float32
        assert live_out.dtype == torch.float32
        assert seen_autocast == [("ema", False), ("live", False)]
    finally:
        h_ema.remove()
        h_live.remove()

    # Bundle save, hypergan sample, and LoadedGenerator all stay FP32
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    save_bundle(bundle_dir, trainer, batch)
    sample_path = sample(bundle_dir, count=4, seed=123, device="cuda:0")
    assert sample_path.is_file()
    loaded = LoadedGenerator(bundle_dir / "model.pt", device="cuda:0")
    assert loaded.graph._bf16_components == frozenset()
    assert loaded.graph._training_bf16_active is False
    loaded_autocast = []
    h_loaded = loaded.graph.models["generator"].register_forward_hook(
        lambda _m, _a, _o: loaded_autocast.append(torch.is_autocast_enabled("cuda"))
    )
    try:
        out = loaded.sample(4, seed=123)
    finally:
        h_loaded.remove()
    assert out.dtype == torch.float32
    assert loaded_autocast == [False]


def test_cuda_bf16_checkpoint_resume_roundtrip_and_mismatch_rejection(tmp_path: Path):
    run_dir = tmp_path / "cuda_bf16_run"
    cfg_bf16 = _conv_model_config(precision="bf16_channels_last", use_bn=False, steps=2)
    cfg_fp32 = _conv_model_config(precision="float32", use_bn=False, steps=2)

    res = train(cfg_bf16, run_dir, steps=2, checkpoint_every=1, stop_after_steps=1)
    assert res["steps"] == 1
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_precision"] == "bf16_channels_last"
    assert manifest["config"]["training"]["precision"] == "bf16_channels_last"

    # Changing precision on resume must be strictly rejected
    with pytest.raises(ValueError, match="Resume configuration differs"):
        resume(run_dir, config_path=cfg_fp32, steps=2)

    # Matching precision resumes cleanly to completion
    resumed = resume(run_dir, config_path=cfg_bf16, steps=2)
    assert resumed["status"] == "complete"
    assert resumed["steps"] == 2


def test_highres_32x32_bf16_skips_simplicial_loss_and_keeps_live_endpoint_reuse():
    cfg = _conv_model_config(precision="bf16_channels_last", use_bn=False, size=32, batch_size=8, steps=2)
    trainer = ReferenceTrainer(cfg)
    t1c = getattr(trainer.gan, "t1c", None) or getattr(trainer.gan, "rhbi", None)
    assert t1c is not None
    simplex_calls = []
    orig_simplex = t1c.simplicial_manifold_loss
    t1c.simplicial_manifold_loss = lambda *a, **kw: (simplex_calls.append(1), orig_simplex(*a, **kw))[1]

    # Verify Upsample receives channels_last inputs and GroupNorm emits channels_last
    # outputs during BF16 training steps, while ema_graph carries zero NHWC hooks.
    from hypergan.recipes import _ChannelsLastInputPreHook, _ChannelsLastOutputHook

    for sub in trainer.ema_graph.modules():
        assert not any(isinstance(fn, _ChannelsLastInputPreHook) for fn in sub._forward_pre_hooks.values())
        assert not any(isinstance(fn, _ChannelsLastOutputHook) for fn in sub._forward_hooks.values())

    upsample_cl = []
    gn_cl = []
    handles = []
    for sub in trainer.graph.models["generator"].modules():
        if isinstance(sub, torch.nn.Upsample):
            handles.append(
                sub.register_forward_hook(
                    lambda _m, inp, _out: upsample_cl.append(inp[0].is_contiguous(memory_format=torch.channels_last))
                )
            )
        elif isinstance(sub, torch.nn.GroupNorm):
            handles.append(
                sub.register_forward_hook(
                    lambda _m, _inp, out: gn_cl.append(out.is_contiguous(memory_format=torch.channels_last))
                )
            )

    trainer.penalty.collect_stats = True
    try:
        row, _ = trainer.update()
    finally:
        for h in handles:
            h.remove()
    assert simplex_calls == [], "3x32x32 (numel=3072 > 1024) must skip simplicial_manifold_loss"
    assert trainer.penalty.last_stats.get("live_endpoints_reused") is True
    assert upsample_cl and all(upsample_cl), f"expected all Upsample inputs NHWC, got {upsample_cl}"
    assert gn_cl and all(gn_cl), f"expected all GroupNorm outputs NHWC, got {gn_cl}"
    assert all(math.isfinite(row[k]) for k in ("d_loss", "g_loss", "gradient_penalty"))


def test_non_particlegan_defaults_hypergan_trains_cleanly_in_bf16_channels_last():
    # Explicit defaults="hypergan" (non-ParticleGAN recipe snapshot) with bf16_channels_last
    raw = {
        "format": "hypergan-model/1",
        "name": "test/cuda-non-particlegan-bf16",
        "defaults": "hypergan",
        "data": {"factory": f"{__name__}:_SyntheticImages", "args": {"channels": 3, "size": 16}},
        "prior": {"kind": "gaussian", "z_dim": 16},
        "networks": {
            "generator": {
                "role": "generator",
                "inputs": {"x": "latent"},
                "input_shape": ["B", 16],
                "output_shape": ["B", 3, 16, 16],
                "source": (
                    "linear(1024)\nreshape(64, 4, 4)\ngroup_norm(8)\nsilu()\n"
                    "upsample(2)\nconv(64, kernel_size=3, padding=1)\ngroup_norm(8)\nsilu()\n"
                    "upsample(2)\nconv(3, kernel_size=3, padding=1)\ntanh()"
                ),
            },
            "discriminator": {
                "role": "critic",
                "inputs": {"x": "candidate"},
                "input_shape": ["B", 3, 16, 16],
                "output_shape": ["B", 1],
                "source": (
                    "conv(64, kernel_size=3, stride=2, padding=1)\nsilu()\n"
                    "conv(64, kernel_size=3, stride=2, padding=1)\nsilu()\n"
                    "flatten()\nlinear(1)"
                ),
            },
        },
        "losses": [{"type": "adversarial", "critic": "discriminator", "weight": 1.0, "penalty": 1.0}],
        "train": {"steps": 2, "batch_size": 8, "seed": 23, "device": "cuda:0", "precision": "bf16_channels_last"},
    }
    cfg = load_config(raw)
    assert cfg["recipe"] == {}
    trainer = ReferenceTrainer(cfg)
    assert trainer.graph._bf16_components == frozenset({"generator", "discriminator"})
    assert trainer.graph._channels_last_components == frozenset({"generator", "discriminator"})
    for _ in range(2):
        row, _ = trainer.update()
        assert all(math.isfinite(row[k]) for k in ("d_loss", "g_loss", "gradient_penalty"))


def test_bf16_detached_real_score_preserves_axiom_transport_capture():
    """Detaching real_score after BF16->FP32 float_output must preserve CaptureEntry for AxiomTransportLoss."""
    # 16x16 (dim=768 <= 1024): AxiomTransportLoss runs on every generator step.
    cfg = _conv_model_config(precision="bf16_channels_last", use_bn=True, size=16, steps=4)
    trainer = ReferenceTrainer(cfg)
    assert trainer.recipe.effective_critic_formulation == "axiom"

    captured_flags = []
    orig_resolve = trainer.gan._resolve_capture_pair

    def _spy_resolve(fake_logits, real_logits):
        fake_entry, real_entry, resolved = orig_resolve(fake_logits, real_logits)
        captured_flags.append(fake_entry is not None and real_entry is not None)
        return fake_entry, real_entry, resolved

    transport_calls = []
    orig_transport = trainer.gan.transport.forward

    def _spy_transport(*args, **kwargs):
        transport_calls.append(len(args))
        return orig_transport(*args, **kwargs)

    trainer.gan._resolve_capture_pair = _spy_resolve
    trainer.gan.transport.forward = _spy_transport

    for _ in range(4):
        row, _ = trainer.update()
        assert all(math.isfinite(row[k]) for k in ("d_loss", "g_loss", "gradient_penalty"))

    # Both d_loss and g_loss on every step must see captured=True, and AxiomTransportLoss must run on all 4 steps.
    assert len(captured_flags) == 8 and all(captured_flags), f"missed captures: {captured_flags}"
    assert len(transport_calls) == 4, f"expected 4 AxiomTransportLoss calls, got {len(transport_calls)}"

    # Continuous 32x32 (dim=3072): captures and projected AxiomTransportLoss both succeed on every step.
    cfg_32 = _conv_model_config(precision="bf16_channels_last", use_bn=True, size=32, steps=2)
    trainer_32 = ReferenceTrainer(cfg_32)
    captured_32 = []
    orig_resolve_32 = trainer_32.gan._resolve_capture_pair

    def _spy_resolve_32(fake_logits, real_logits):
        fake_entry, real_entry, resolved = orig_resolve_32(fake_logits, real_logits)
        captured_32.append(fake_entry is not None and real_entry is not None)
        return fake_entry, real_entry, resolved

    transport_32 = []
    orig_transport_32 = trainer_32.gan.transport.forward

    def _spy_transport_32(*args, **kwargs):
        transport_32.append(len(args))
        return orig_transport_32(*args, **kwargs)

    trainer_32.gan._resolve_capture_pair = _spy_resolve_32
    trainer_32.gan.transport.forward = _spy_transport_32

    for _ in range(2):
        row, _ = trainer_32.update()
        assert all(math.isfinite(row[k]) for k in ("d_loss", "g_loss", "gradient_penalty"))

    assert len(captured_32) == 4 and all(captured_32), f"missed 32x32 captures: {captured_32}"
    assert len(transport_32) == 2, f"expected 2 32x32 AxiomTransportLoss calls, got {len(transport_32)}"
