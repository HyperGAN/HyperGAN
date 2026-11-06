"""Comprehensive unit tests covering Milestone 3 features (Features 25–39)."""
import io
import json
from pathlib import Path
import pytest
import torch

import hypergan as hg
from hypergan.network_config import validate_network_args
from hypergan.pretrained_providers import _resnet18_shape_fn, _dinov3_shape_fn
from hypergan.model_file import ModelFileError, lower, require_networks
from hypergan.schema_info import get_schema
from hypergan.ops_info import get_operators
from hypergan.run_info import RunInfo, read_run_info
from hypergan.summary import write_summary_json, read_summary_json
from hypergan.bounded_cli_output import CLIProgress
from hypergan.metrics import DEFAULT_METRICS, metric_catalog, select_metrics
from hypergan.config import resolve_config


def test_capability_probes_active():
    import inspect
    from hypergan.cli import _build_parser
    parser = _build_parser()
    assert len(inspect.signature(validate_network_args).parameters) >= 1
    assert hasattr(hg, "schema")
    assert hasattr(hg, "operators")
    assert hasattr(hg, "read_run_info")
    assert hasattr(hg, "RunInfo")
    assert "--json" in parser._option_string_actions
    assert "--quiet" in parser._option_string_actions


def test_feature_25_static_hndl_ast_validation():
    # Valid specification
    valid_spec = {
        "source": "h = linear(x, 16)\nlinear(h, 2)",
        "input_shape": ["B", 4],
        "output_shape": ["B", 2],
    }
    assert validate_network_args(valid_spec) == []

    # Invalid specification with syntax/resolution error
    invalid_spec = {
        "source": "unknown_operator_xyz(123)",
        "input_shape": ["B", 4],
        "output_shape": ["B", 2],
    }
    errors = validate_network_args(invalid_spec)
    assert len(errors) > 0
    assert any("unknown_operator_xyz" in err for err in errors)

    # Legacy location behavior raises ValueError on structural error
    with pytest.raises(ValueError):
        validate_network_args({"source": "foo", "unknown_arg": 1}, location="networks.generator")


def test_feature_26_symbolic_provider_shape_functions():
    # ResNet-18 shape function
    r18_out = _resnet18_shape_fn([16, 3, 64, 64], layer="layer2")
    assert r18_out == (16, 128, 8, 8)

    # DINOv3 shape function
    dino_out = _dinov3_shape_fn([8, 3, 224, 224], readout="patch_tokens")
    assert dino_out == (8, 196, 384)


def test_feature_27_model_file_typo_suggestions():
    # Unknown table suggestion: [tran] -> [train]
    with pytest.raises(ModelFileError) as exc_info:
        require_networks({"tran": {"batch_size": 16}})
    assert "did you mean '[train]'?" in str(exc_info.value).lower()


def test_feature_28_and_29_schema_and_ops_catalog():
    # Schema info
    full_schema = get_schema()
    assert full_schema["format"] == "hypergan-model/1"
    assert "sections" in full_schema
    assert "train" in full_schema["sections"]
    assert "networks" in full_schema["sections"]

    # Filtered schema section
    train_schema = get_schema("train")
    assert "fields" in train_schema

    # Ops info (all 85 operators across categories)
    all_ops = get_operators()
    assert len(all_ops) == 85
    op_names = {op["name"] for op in all_ops}
    assert "conv" in op_names
    assert "linear" in op_names
    assert "relu" in op_names
    assert "adaptive_norm" in op_names

    # Filtered ops query
    norm_ops = get_operators("norm")
    assert len(norm_ops) > 0
    assert all("norm" in (op["name"] + op["category"] + op.get("summary", "") + op.get("doc", "")).lower() for op in norm_ops)


def test_feature_30_docs_presence():
    root = Path(__file__).resolve().parent.parent.parent
    llms_txt = root / "llms.txt"
    hndl_ref = root / "docs" / "hndl-reference.md"

    assert llms_txt.is_file()
    assert "HyperGAN" in llms_txt.read_text(encoding="utf-8")

    assert hndl_ref.is_file()
    assert "HNDL" in hndl_ref.read_text(encoding="utf-8")


def test_feature_34_and_35_cadence_defaults_and_restoration():
    assert DEFAULT_METRICS["every_steps"] == 10

    # Resolved configuration inherits metrics.every_steps = 10
    resolved = resolve_config({})
    assert resolved["metrics"]["every_steps"] == 10

    # Model lowering of packaged template preserves metrics.every_steps = 10
    m = hg.packaged("gaussian-grid")
    lowered = lower(m.spec)
    assert lowered["metrics"]["every_steps"] == 10


def test_feature_36_progress_line_loss_rendering():
    class MockOutput:
        quiet = False
        progress_json = False
        evaluation_reminder = None
        def __init__(self):
            self.stderr = io.StringIO()
            class Policy:
                every = 1
                @staticmethod
                def refresh(stream): pass
            self.policy = Policy()

    mock = MockOutput()
    progress = CLIProgress(mock)
    event = {
        "event": "train",
        "step": 50,
        "total_steps": 100,
        "steps_per_second": 10.5,
        "d_loss": 0.6931,
        "g_loss": 0.7012,
        "metrics": {"loss/d_total": 0.6931, "loss/g_total": 0.7012},
    }
    progress(event)
    rendered = mock.stderr.getvalue()
    assert "D=0.6931" in rendered
    assert "G=0.7012" in rendered


def test_feature_37_health_telemetry_catalog_registration():
    config = {
        "metrics": {"preset": "standard", "every_steps": 10, "disable": [], "overrides": {}, "custom": {}},
        "objectives": [],
        "adversarial": {"weight": 1.0},
        "gradient_penalty": {"lazy_k": 1, "coeff": 1.0},
        "prior_regularizer": {"weight": 1.0},
    }
    catalog = metric_catalog(config)
    telemetry_keys = [
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
    ]
    for key in telemetry_keys:
        assert key in catalog["metrics"]


def test_feature_38_telemetry_trends_and_health_alerts(tmp_path):
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    manifest = {
        "run_id": "test_alerts",
        "status": "running",
        "steps": 25,
        "config": {"training": {"steps": 100}},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Metrics with vanishing critic gradient and low diversity
    metrics = {
        "loss/d_total": 0.5,
        "loss/g_total": 1.2,
        "grad_norm/critic": 1e-6,
        "diversity/ratio": 0.15,
        "critic/accuracy": 0.99,
    }
    write_summary_json(run_dir, manifest, metrics=metrics)
    summary = read_summary_json(run_dir)

    assert "telemetry_trends" in summary
    assert "loss/d_total" in summary["telemetry_trends"]

    alerts = summary.get("health_alerts", [])
    assert len(alerts) >= 2
    assert any("vanishing gradient" in a.lower() for a in alerts)
    assert any("mode collapse" in a.lower() for a in alerts)
    assert any("overpowering" in a.lower() for a in alerts)


def test_feature_39_unified_run_info_reader(tmp_path):
    run_dir = tmp_path / "test_run_info"
    run_dir.mkdir()
    manifest = {
        "run_id": "0123456789abcdef0123456789abcdef",
        "status": "complete",
        "steps": 100,
        "config": {"name": "test-exp", "training": {"steps": 100}},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    summary_data = {
        "run_id": manifest["run_id"],
        "status": "complete",
        "step": 100,
        "total_steps": 100,
        "latest": {"loss/d_total": 0.45},
        "health_alerts": [],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_data), encoding="utf-8")

    info = read_run_info(run_dir)
    assert isinstance(info, RunInfo)
    assert info.run_id == manifest["run_id"]
    assert info.step == 100
    assert info.total_steps == 100
    # Normalized status
    assert info.status == "completed"
    assert info.latest_metrics == {"loss/d_total": 0.45}
    assert info.name == "test-exp"
    assert info.short_id == "01234567"
    assert isinstance(info.to_dict(), dict)
