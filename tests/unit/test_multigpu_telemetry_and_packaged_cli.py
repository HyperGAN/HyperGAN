"""Tests for 2+ GPU ReplicatedTrainer telemetry parity, distributed-checkpoints summary/status,
direct packaged recipe CLI/API resolution, and self-contained cifar10-transgan cache paths.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

import pytest
import torch
import torch.distributed as dist

import hypergan as hg
from hypergan import cli
from hypergan.config import PACKAGED_MODELS_DIR, load_config, resolve_config, resolve_packaged_model_path
from hypergan.cpu_workers import launch_cpu_workers
from hypergan.distributed_training import ReplicatedCPUTrainer
from hypergan.metrics import validate_update_scalars
from hypergan.run_info import read_run_info
from hypergan.summary import collect_checkpoints, read_summary_json, write_summary_json
from hypergan.training import ReferenceTrainer
from tests.config_fixtures import write_default


CANONICAL_HEALTH_KEYS = (
    "critic/real_mean",
    "critic/fake_mean",
    "critic/margin",
    "critic/accuracy",
    "grad_norm/critic",
    "grad_norm/generator",
    "grad_norm/prior",
    "optimizer/d_guard_ratio",
    "prior/particle_rms",
    "diversity/ratio",
)

SNAKE_HEALTH_KEYS = (
    "critic_real_mean",
    "critic_fake_mean",
    "critic_margin",
    "critic_accuracy",
    "grad_norm_critic",
    "grad_norm_generator",
    "grad_norm_prior",
    "d_guard_ratio",
    "particle_rms",
    "diversity_ratio",
)

ALL_TELEMETRY_KEYS = CANONICAL_HEALTH_KEYS + SNAKE_HEALTH_KEYS


def _encoded_json(row: dict) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _telemetry_parity_worker(rank, world_size, config_dict, batches, shared_dir, accumulation_steps):
    config = resolve_config(config_dict)
    trainer = ReplicatedCPUTrainer(config, world_size=world_size, accumulation_steps=accumulation_steps)

    step_outputs = []
    all_reduce_counts = []
    orig_all_reduce = dist.all_reduce

    for full_batch in batches:
        local_batch = {k: v.chunk(world_size)[rank] for k, v in full_batch.items() if not k.startswith("_")}
        ids = full_batch["_ids"].chunk(world_size)[rank]
        prior_arg = (trainer.prior.z[ids], ids)

        calls = 0
        f64_calls = 0

        def counted_all_reduce(tensor, *args, **kwargs):
            nonlocal calls, f64_calls
            calls += 1
            if getattr(tensor, "dtype", None) == torch.float64:
                f64_calls += 1
            return orig_all_reduce(tensor, *args, **kwargs)

        dist.all_reduce = counted_all_reduce
        try:
            metrics, _ = trainer.update(local_batch, prior_arg)
        finally:
            dist.all_reduce = orig_all_reduce

        step_outputs.append(metrics)
        all_reduce_counts.append({"total": calls, "float64_metric_collectives": f64_calls})

    out_path = Path(shared_dir) / f"rank-{rank}.json"
    out_path.write_text(
        json.dumps({"steps": step_outputs, "all_reduce_counts": all_reduce_counts}),
        encoding="utf-8",
    )


@pytest.mark.parametrize("accumulation_steps", [1, 2])
def test_replicated_trainer_telemetry_parity_and_rank_agreement(tmp_path, accumulation_steps):
    """Fix 1: ReplicatedTrainer emits all 10 GAN health scalars (+ snake_case aliases) with
    bit-for-bit rank agreement, ReferenceTrainer parity, and zero extra dist.all_reduce calls.
    """
    config_dict = {
        "prior": {"kind": "mog", "fixed_sigma": 0.05, "args": {"num_particles": 16, "z_dim": 4}},
        "components": {
            "generator": {
                "factory": "linear",
                "args": {"in_features": 4, "out_features": 2, "bias": True},
                "inputs": {"input": "latent"},
            },
            "discriminator": {
                "factory": "linear",
                "args": {"in_features": 2, "out_features": 1, "bias": True},
                "inputs": {"input": "candidate"},
            },
        },
        "optimizer": {"lr": 1e-3},
        "gradient_penalty": {"coeff": 0.5, "lazy_k": 1},
        "prior_regularizer": {"weight": 0.05, "rows": "full"},
        "training": {"steps": 4, "batch_size": 8, "ema": 0.99, "device": "cpu", "seed": 1234},
    }

    torch.manual_seed(77)
    batches = [
        {"real": torch.randn(8, 2), "_ids": torch.arange(0, 8, dtype=torch.long)},
        {"real": torch.randn(8, 2), "_ids": torch.arange(8, 16, dtype=torch.long)},
    ]

    # Single-process ReferenceTrainer oracle
    ref_trainer = ReferenceTrainer(resolve_config(copy.deepcopy(config_dict)))
    ref_steps = []
    for full_batch in batches:
        ids = full_batch["_ids"]
        ref_metrics, _ = ref_trainer.update({"real": full_batch["real"]}, (ref_trainer.prior.z[ids], ids))
        ref_steps.append(ref_metrics)

    # 2-rank Gloo ReplicatedCPUTrainer
    launch_cpu_workers(
        _telemetry_parity_worker,
        args=(config_dict, batches, str(tmp_path), accumulation_steps),
        world_size=2,
        timeout=30,
        collective_timeout=10,
    )

    rank0 = json.loads((tmp_path / "rank-0.json").read_text(encoding="utf-8"))
    rank1 = json.loads((tmp_path / "rank-1.json").read_text(encoding="utf-8"))

    # Steady-state step 2 uses exactly 5 dist.all_reduce calls under gradient_allreduce:
    # (control + bucket for D) + (control + bucket for G) + 1 coalesced float64 metric scalar all_reduce.
    # Zero extra round-trips for the 10 health telemetry scalars!
    assert rank0["all_reduce_counts"][1] == {"total": 5, "float64_metric_collectives": 1}
    assert rank1["all_reduce_counts"][1] == {"total": 5, "float64_metric_collectives": 1}

    for step_idx in range(len(batches)):
        m0 = rank0["steps"][step_idx]
        m1 = rank1["steps"][step_idx]
        m_ref = ref_steps[step_idx]

        # Bit-for-bit JSON agreement across ranks (required by ReplicatedExecution.update)
        assert _encoded_json(m0) == _encoded_json(m1)
        validate_update_scalars(m0, objective_count=0)

        for key in ALL_TELEMETRY_KEYS:
            assert key in m0, f"Missing telemetry key {key}"
            assert isinstance(m0[key], float) and math.isfinite(m0[key])

        for canonical, snake in zip(CANONICAL_HEALTH_KEYS, SNAKE_HEALTH_KEYS):
            assert m0[canonical] == m0[snake]
            assert m0[canonical] == pytest.approx(m_ref[canonical], rel=1e-4, abs=1e-5), (
                f"Step {step_idx + 1} mismatch for {canonical}: replicated={m0[canonical]} vs ref={m_ref[canonical]}"
            )


def test_validate_update_scalars_rejects_invalid_telemetry():
    """Fix 1: validate_update_scalars enforces finite numeric values for all telemetry keys."""
    valid_row = {
        "d_loss": 1.0,
        "g_loss": 1.0,
        "d_adversarial": 0.8,
        "g_adversarial": 0.9,
        "prior_loss": 0.1,
        "gradient_penalty": 0.2,
        "lr_scale": 1.0,
        "d_adversarial_weighted": 0.8,
        "g_adversarial_weighted": 0.9,
        "objectives": [],
    }
    for key in ALL_TELEMETRY_KEYS:
        valid_row[key] = 0.5
    validate_update_scalars(valid_row, objective_count=0)

    for bad_val in (float("nan"), float("inf"), -float("inf"), True, "0.5"):
        for key in ("critic/real_mean", "grad_norm/generator", "diversity/ratio", "d_guard_ratio"):
            bad_row = dict(valid_row, **{key: bad_val})
            with pytest.raises(ValueError, match="Invalid finite global metric"):
                validate_update_scalars(bad_row, objective_count=0)


REPLICATED_E2E_DRIVER = """
from pathlib import Path
import sys
from hypergan.replicated_execution import run_train

if __name__ == "__main__":
    config_path = Path(sys.argv[1])
    run_dir = Path(sys.argv[2])
    profile = {
        "schema_version": 1,
        "execution": {"name": "cpu-replicated-gloo", "world_size": 2, "accumulation_steps": 1},
    }
    run_train(config_path, run_dir, checkpoint_every=10, profile=profile)
"""


def test_cpu_replicated_gloo_end_to_end_summary_telemetry_and_distributed_checkpoints(tmp_path, capsys):
    """Fixes 1 & 2: 2-rank cpu-replicated-gloo run writes all 10 health scalars, telemetry_trends,
    health_alerts, and distributed-checkpoints into summary.json and `hypergan status`.
    """
    cfg_path = write_default(tmp_path / "replicated20.toml", device="cpu", defaults="hypergan")
    cfg_text = (
        cfg_path.read_text(encoding="utf-8")
        .replace("steps = 5", "steps = 20")
        .replace("num_particles = 20000", "num_particles = 32")
        .replace("count = 256", "count = 16")
    )
    cfg_path.write_text(cfg_text + '\n[metrics]\nevery_steps = 10\n', encoding="utf-8")

    driver = tmp_path / "run_replicated_driver.py"
    driver.write_text(REPLICATED_E2E_DRIVER, encoding="utf-8")
    run_dir = tmp_path / "run-2rank"

    src_root = str(Path(hg.__file__).resolve().parents[1])
    existing_pp = os.environ.get("PYTHONPATH", "")
    py_path = f"{src_root}{os.pathsep}{existing_pp}" if existing_pp else src_root
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=py_path)
    result = subprocess.run(
        [sys.executable, str(driver), str(cfg_path), str(run_dir)],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    summary = read_summary_json(run_dir)
    assert summary is not None
    assert summary["step"] == 20

    # Fix 1: All 10 GAN health scalars present in summary.json["latest"] and telemetry_trends
    for key in CANONICAL_HEALTH_KEYS:
        assert key in summary["latest"], f"Missing {key} in summary['latest']"
        assert isinstance(summary["latest"][key], (int, float)) and math.isfinite(summary["latest"][key])
        assert key in summary["telemetry_trends"], f"Missing {key} in summary['telemetry_trends']"
        trend = summary["telemetry_trends"][key]
        assert isinstance(trend, (int, float)) and math.isfinite(trend)

    assert isinstance(summary["health_alerts"], list)

    # Fix 2: distributed-checkpoints/ populated into summary.json["checkpoints"] and `hypergan status`
    assert (run_dir / "distributed-checkpoints").is_dir()
    assert not (run_dir / "checkpoints").exists()
    ckpts = summary["checkpoints"]
    assert [c["step"] for c in ckpts] == [0, 10, 20]
    assert all(c["path"].startswith("distributed-checkpoints/") for c in ckpts)

    capsys.readouterr()
    rc = cli.main(["status", str(run_dir)])
    assert rc == 0
    rendered = capsys.readouterr().out
    assert "Recent Checkpoints" in rendered
    assert "10" in rendered and "20" in rendered

    rc_json = cli.main(["status", str(run_dir), "--json"])
    assert rc_json == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert sorted(c["step"] for c in status_payload["recent_checkpoints"]) == [0, 10, 20]


def test_collect_checkpoints_scans_both_checkpoints_and_distributed_checkpoints(tmp_path):
    """Fix 2: collect_checkpoints, write_summary_json, read_run_info, and `hypergan status`
    scan both checkpoints/ and distributed-checkpoints/, skip hidden dirs (.prepared, .pending-*),
    and deduplicate/sort by (step, timestamp).
    """
    run_dir = tmp_path / "cifar10-transgan-2gpu"
    dist_root = run_dir / "distributed-checkpoints"
    dist_root.mkdir(parents=True)

    # Hidden staging dirs and latest.json pointer file must be ignored
    (dist_root / ".prepared" / "staging-1").mkdir(parents=True)
    (dist_root / ".prepared" / "staging-1" / "manifest.json").write_text(json.dumps({"step": 999}), encoding="utf-8")
    (dist_root / ".pending-step-500").mkdir()
    (dist_root / ".pending-step-500" / "manifest.json").write_text(json.dumps({"step": 500}), encoding="utf-8")
    (dist_root / "latest.json").write_text(json.dumps({"step": 1000}), encoding="utf-8")

    # Distributed snapshots at steps 0, 53, and 1000 (mirrors live 2-GPU validation scenario)
    for step, ts in (
        (1000, "2026-10-30T18:10:00+00:00"),
        (0, "2026-10-30T18:00:00+00:00"),
        (53, "2026-10-30T18:02:00+00:00"),
    ):
        snap = dist_root / f"attempt1-step-{step:08d}-abcdef012345"
        snap.mkdir()
        (snap / "manifest.json").write_text(
            json.dumps({"step": step, "created_at": ts}),
            encoding="utf-8",
        )

    ckpts = collect_checkpoints(run_dir)
    assert [c["step"] for c in ckpts] == [0, 53, 1000]
    assert all(c["path"].startswith("distributed-checkpoints/") for c in ckpts)

    # Even if an older summary.json wrote `"checkpoints": []`, `hg.status` & `read_run_info` recover from disk
    manifest = {
        "run_id": "run_2gpu_test",
        "status": "running",
        "steps": 1000,
        "config": {"name": "cifar10-transgan-2gpu", "training": {"steps": 10000}},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "run_2gpu_test",
                "name": "cifar10-transgan-2gpu",
                "status": "running",
                "step": 1000,
                "total_steps": 10000,
                "checkpoints": [],
                "evaluations": [],
                "latest": {},
            }
        ),
        encoding="utf-8",
    )

    info = read_run_info(run_dir)
    assert [c["step"] for c in info.checkpoints] == [0, 53, 1000]

    status_dict = hg.status(run_dir)
    assert sorted(c["step"] for c in status_dict["recent_checkpoints"]) == [0, 53, 1000]

    # Re-writing summary.json populates summary["checkpoints"] cleanly
    write_summary_json(run_dir, manifest)
    refreshed = read_summary_json(run_dir)
    assert [c["step"] for c in refreshed["checkpoints"]] == [0, 53, 1000]

    # Mixed run: add single-GPU checkpoints/ (including a directory-name fallback & hidden .pending dir)
    single_root = run_dir / "checkpoints"
    (single_root / ".pending-25").mkdir(parents=True)
    (single_root / ".pending-25" / "manifest.json").write_text(json.dumps({"step": 25}), encoding="utf-8")

    s25 = single_root / "0001-step-0000025-112233"
    s25.mkdir(parents=True)
    (s25 / "manifest.json").write_text(
        json.dumps({"step": 25, "created_at": "2026-10-30T18:01:00+00:00"}),
        encoding="utf-8",
    )

    # Snapshot without manifest step falls back to directory-name step parsing
    s1500 = dist_root / "attempt2-step-00001500-fedcba987654"
    s1500.mkdir()

    mixed = collect_checkpoints(run_dir)
    assert [c["step"] for c in mixed] == [0, 25, 53, 1000, 1500]
    by_step = {c["step"]: c for c in mixed}
    assert by_step[25]["path"] == "checkpoints/0001-step-0000025-112233"
    assert by_step[53]["path"] == "distributed-checkpoints/attempt1-step-00000053-abcdef012345"
    assert by_step[1500]["path"] == "distributed-checkpoints/attempt2-step-00001500-fedcba987654"


def test_packaged_recipe_name_resolution_across_cli_and_api(tmp_path, monkeypatch, capsys):
    """Fix 3: Bare packaged model names (with or without .toml) resolve across hg.load,
    load_config, validate, preflight, data-check, queue add, explain, and train.
    """
    empty_cwd = tmp_path / "empty_workspace"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)
    monkeypatch.setenv("HYPERGAN_HOME", str(tmp_path / "hg_home"))

    expected_transgan = (PACKAGED_MODELS_DIR / "cifar10-transgan.toml").resolve()
    assert resolve_packaged_model_path("cifar10-transgan") == expected_transgan
    assert resolve_packaged_model_path("cifar10-transgan.toml") == expected_transgan

    # hg.load & load_config
    m1 = hg.load("cifar10-transgan")
    m2 = hg.load("cifar10-transgan.toml")
    assert m1.path == expected_transgan
    assert m2.path == expected_transgan
    cfg = load_config("cifar10-transgan")
    assert cfg["data"]["factory"] == "hypergan.image_data:CIFAR10Data"

    # hypergan validate (plain & --json)
    capsys.readouterr()
    assert cli.main(["validate", "cifar10-transgan"]) == 0
    assert "valid" in capsys.readouterr().out.lower()

    assert cli.main(["validate", "cifar10-transgan", "--json"]) == 0
    v_payload = json.loads(capsys.readouterr().out)
    assert v_payload["data"]["factory"] == "hypergan.image_data:CIFAR10Data"
    assert v_payload["training"]["steps"] == 200000

    # hypergan explain
    assert cli.main(["explain", "cifar10-transgan", "--json"]) == 0
    exp_payload = json.loads(capsys.readouterr().out)
    assert Path(exp_payload["path"]).resolve() == expected_transgan

    # hypergan preflight
    assert cli.main(["preflight", "cifar10-transgan", "--json"]) == 0
    pf_payload = json.loads(capsys.readouterr().out)
    assert pf_payload["stage"] == "structural"

    # hypergan data-check (both Python API and CLI): resolves packaged config instead of FileNotFoundError
    with pytest.raises(ValueError, match="image_folder"):
        hg.data_check("cifar10-transgan")
    assert cli.main(["data-check", "cifar10-transgan"]) != 0
    dc_captured = capsys.readouterr()
    assert "image_folder" in (dc_captured.out + dc_captured.err)
    assert "No such file or directory" not in (dc_captured.out + dc_captured.err)

    # hypergan queue add
    queue_file = tmp_path / "queue.json"
    assert cli.main(["queue", "add", "cifar10-transgan", "--queue-file", str(queue_file)]) == 0
    capsys.readouterr()
    queued = json.loads(queue_file.read_text(encoding="utf-8"))
    assert Path(queued["jobs"][0]["model_file"]).resolve() == expected_transgan

    # hypergan train <packaged-name>
    train_run_dir = tmp_path / "gaussian-grid-cli-run"
    rc_train = cli.main(
        [
            "train",
            "gaussian-grid",
            "--devices",
            "cpu",
            "--steps",
            "2",
            "--no-server",
            "--run-dir",
            str(train_run_dir),
        ]
    )
    train_captured = capsys.readouterr()
    assert rc_train == 0, train_captured.out + train_captured.err
    trained_summary = read_summary_json(train_run_dir)
    assert trained_summary is not None
    assert trained_summary["step"] == 2


def test_local_files_and_existing_runs_take_precedence_over_packaged_names(tmp_path, monkeypatch):
    """Fix 3: Local files and existing run directories/slugs ALWAYS win over packaged model names."""
    ws = tmp_path / "precedence_ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    monkeypatch.setenv("HYPERGAN_HOME", str(tmp_path / "hg_home_prec"))

    # 1. Local file `./gaussian-grid.toml` overrides packaged `gaussian-grid.toml`
    packaged_gg = (PACKAGED_MODELS_DIR / "gaussian-grid.toml").read_text(encoding="utf-8")
    local_gg = ws / "gaussian-grid.toml"
    local_gg.write_text(re.sub(r"steps\s*=\s*\d+", "steps = 77", packaged_gg), encoding="utf-8")

    assert resolve_packaged_model_path("gaussian-grid.toml") is None
    loaded_local = hg.load("gaussian-grid.toml")
    assert loaded_local.path == local_gg.resolve()
    assert hg.resolve(loaded_local)["training"]["steps"] == 77

    # 2. Local directory `./gaussian-grid/` with model.toml overrides bare `gaussian-grid`
    local_run_dir = ws / "gaussian-grid"
    local_run_dir.mkdir()
    (local_run_dir / "model.toml").write_text(
        re.sub(r"steps\s*=\s*\d+", "steps = 42", packaged_gg),
        encoding="utf-8",
    )
    assert resolve_packaged_model_path("gaussian-grid") is None
    loaded_dir = hg.load("gaussian-grid")
    assert hg.resolve(loaded_dir)["training"]["steps"] == 42

    # 3. Unknown bare name still raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        hg.load("nonexistent-packaged-recipe-xyz")


def test_cifar10_transgan_uses_self_contained_hypergan_cache_paths(tmp_path, monkeypatch):
    """Fix 4: cifar10-transgan.toml uses builtin='cifar10', root='~/.hypergan/cache/cifar10',
    and HTTPS Inception weights URL + sha256 matching cifar10-tiny-transformer.toml.
    """
    empty_dir = tmp_path / "fresh_dir"
    empty_dir.mkdir()
    monkeypatch.chdir(empty_dir)

    transgan_path = PACKAGED_MODELS_DIR / "cifar10-transgan.toml"
    tiny_path = PACKAGED_MODELS_DIR / "cifar10-tiny-transformer.toml"

    raw_transgan = tomllib.loads(transgan_path.read_text(encoding="utf-8"))
    raw_tiny = tomllib.loads(tiny_path.read_text(encoding="utf-8"))

    assert raw_transgan["data"]["builtin"] == "cifar10"
    assert raw_transgan["data"]["args"]["root"] == "~/.hypergan/cache/cifar10"
    assert raw_transgan["data"]["args"]["root"] == raw_tiny["data"]["args"]["root"]

    eval_transgan = raw_transgan["evaluations"]["fid50k_train"]
    eval_tiny = raw_tiny["evaluations"]["fid50k_train"]
    assert eval_transgan["data"]["builtin"] == "cifar10"
    assert eval_transgan["data"]["args"]["root"] == "~/.hypergan/cache/cifar10"
    assert eval_transgan["args"]["weights_path"] == (
        "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/"
        "weights-inception-2015-12-05-6726825d.pth"
    )
    assert eval_transgan["args"]["weights_sha256"] == (
        "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"
    )
    assert eval_transgan["args"]["weights_path"] == eval_tiny["args"]["weights_path"]
    assert eval_transgan["args"]["weights_sha256"] == eval_tiny["args"]["weights_sha256"]

    warnings = hg.validate("cifar10-transgan")
    assert isinstance(warnings, list)


def test_cifar10_transgan_uses_lazy_k_1_and_particlegan_spike_guard_settings():
    """cifar10-transgan uses lazy_k=1 and ParticleGAN spike guard defaults (5.0 / 200)."""
    cfg = load_config("cifar10-transgan")
    assert cfg["gradient_penalty"]["lazy_k"] == 1
    assert cfg["optimizer"]["d_guard_ratio"] == 5.0
    assert cfg["optimizer"]["d_guard_min_steps"] == 200


def test_ema_critic_is_initialized_in_eval_mode_and_d_guard_ratio_tracks_spike_clipping():
    """ema_critic is placed in eval() mode (so stochastic DiffAugment/Dropout do not fire on the anchor)
    and optimizer/d_guard_ratio reports the actual post-guard / pre-guard gradient norm ratio (< 1.0 on spikes).
    """
    from hypergan.config import resolve_config
    from hypergan.training import ReferenceTrainer

    cfg = resolve_config({
        "gradient_penalty": {"anchor_weight": 1.0, "lazy_k": 1},
        "optimizer": {"d_guard_ratio": 5.0, "d_guard_min_steps": 1},
        "training": {"steps": 10, "batch_size": 8},
        "prior": {"args": {"num_particles": 32, "z_dim": 4}},
    })
    trainer = ReferenceTrainer(cfg)
    assert trainer.critic.training is True
    assert trainer.opt_d.ema_critic is not None
    assert trainer.opt_d.ema_critic.training is False

    # Warm up 2 normal steps so Adam exp_avg_sq is populated with small gradient scale
    row1, _ = trainer.update()
    assert row1["optimizer/d_guard_ratio"] == pytest.approx(1.0)
    row2, _ = trainer.update()
    assert 0.0 < row2["optimizer/d_guard_ratio"] <= 1.0

    # Simulate Adam second-moment (v_t) decay during a prolonged flat-margin plateau,
    # followed by a normal gradient step (> 5x sqrt(vhat)) so CriticSpikeGuard clips.
    for st in trainer.opt_d.state.values():
        if "exp_avg_sq" in st:
            st["exp_avg_sq"].mul_(1e-6)
    row_spike, _ = trainer.update()
    assert 0.0 < row_spike["optimizer/d_guard_ratio"] < 0.2
    assert trainer.opt_d.guard.clipped_tensors > 0


def test_cli_train_packaged_recipe_does_not_hijack_existing_run_with_same_experiment_name(tmp_path, monkeypatch):
    """Running `hypergan train gaussian-grid --run-dir ...` (or a second `hypergan train gaussian-grid`)
    must create a new run from the packaged recipe rather than hijacking an older run whose
    manifest config.name is 'gaussian-grid'.
    """
    monkeypatch.chdir(tmp_path)
    run1 = tmp_path / "runs" / "first-run"
    run2 = tmp_path / "runs" / "second-run"

    rc1 = cli.main([
        "train", "gaussian-grid",
        "--run-dir", str(run1),
        "-d", "cpu",
        "--steps", "2",
        "--no-previews",
        "--no-server",
        "-q",
    ])
    assert rc1 == 0
    assert (run1 / "manifest.json").is_file()

    rc2 = cli.main([
        "train", "gaussian-grid",
        "--run-dir", str(run2),
        "-d", "cpu",
        "--steps", "2",
        "--no-previews",
        "--no-server",
        "-q",
    ])
    assert rc2 == 0
    assert (run2 / "manifest.json").is_file()
    m1 = json.loads((run1 / "manifest.json").read_text(encoding="utf-8"))
    m2 = json.loads((run2 / "manifest.json").read_text(encoding="utf-8"))
    assert m1["run_id"] != m2["run_id"]
    assert m1["steps"] == 2
    assert m2["steps"] == 2
