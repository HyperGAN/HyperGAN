"""Follow-ups to the 2026-10-25 review of the CLI, Python API and model-file layer."""

import inspect
import io
import json
from pathlib import Path

import pytest

import hypergan
import hypergan.api as hg
from hypergan import cli
from hypergan.model_file import (
    DEFAULT_CHECKPOINT_POLICY, ModelFileError, collect_source_files, lower, normalize_checkpoint_policy,
    resolve_inheritance,
)

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "api" / "config_first"


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERGAN_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("HYPERGAN_STATE_DIR", raising=False)
    monkeypatch.delenv("HYPERGAN_WORKSPACE", raising=False)
    monkeypatch.delenv("HYPERGAN_DEBUG", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(directory, run_id="c0ffee00c0ffee00", summary=None):
    run = Path(directory)
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"run_id": run_id, "status": "completed"}), encoding="utf-8")
    if summary is not None:
        (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run


def _subparsers(parser):
    import argparse
    return next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices


# --------------------------------------------------------------------------- run-target help, prune alias


def test_run_management_help_names_every_accepted_target():
    commands = _subparsers(cli._parser())
    parsers = [commands[name] for name in ("status", "logs", "stop", "tag", "rm", "gc", "checkpoint")]
    parsers.append(_subparsers(commands["checkpoints"])["prune"])
    for parser in parsers:
        action = next(a for a in parser._actions if a.dest == "run_dir")
        assert "run ID, short ID, slug, tag, run directory name, or path" in action.help, parser.prog
        assert "or name" not in action.help


def test_prune_is_a_hidden_deprecated_spelling_of_gc(isolated, monkeypatch, capsys):
    commands = _subparsers(cli._parser())
    assert "prune" not in commands and "gc" in commands
    seen = []
    monkeypatch.setattr(hg, "gc", lambda run, **kwargs: seen.append((run, kwargs)) or {"run_dir": str(run)})
    assert cli.main(["prune", "my-run", "--dry-run"]) in (0, None)
    err = capsys.readouterr().err
    assert "`hypergan prune` is deprecated" in err
    assert "hypergan gc" in err and "hypergan checkpoints prune" in err
    assert seen and seen[0][0] == Path("my-run") and seen[0][1]["dry_run"] is True


def test_only_the_top_level_command_is_mapped(isolated):
    # A run named "prune" is an ordinary argument of a subcommand.
    args = cli._parser().parse_args(["tag", "prune", "newname"])
    assert (args.command, args.run_dir, args.name) == ("tag", Path("prune"), "newname")
    args = cli._parser().parse_args(["--json", "prune", "r"])
    assert args.command == "gc"


def test_singular_checkpoint_prune_names_the_right_command(isolated, capsys):
    with pytest.raises(SystemExit) as exc:
        cli._parser().parse_args(["checkpoint", "prune", "run", "--dry-run"])
    assert exc.value.code == 2
    assert "use `hypergan checkpoints prune`" in capsys.readouterr().err


def test_parser_aliases_are_gone():
    assert not hasattr(cli, "_build_parser") and not hasattr(cli, "_add_run_options")


# --------------------------------------------------------------------------- checkpoints prune resolution


def test_prune_checkpoints_resolves_strictly(isolated, monkeypatch):
    import hypergan.checkpoints as checkpoints
    seen = []
    monkeypatch.setattr(checkpoints, "prune_checkpoints", lambda run, **kwargs: seen.append(run) or {"run_dir": str(run)})
    run = _run(isolated / "runs" / "cifar-baseline")
    hg.prune_checkpoints(run, dry_run=True)
    assert seen == [run.resolve()]
    # A directory without a run identity (here the parent of runs) is refused,
    # where the fuzzy read-only resolver returned the raw path.
    with pytest.raises(ValueError, match="Refusing to act"):
        hg.prune_checkpoints(isolated / "runs")
    (isolated / "plain").mkdir()
    with pytest.raises(ValueError, match="not a HyperGAN run directory"):
        hg.prune_checkpoints(isolated / "plain")
    assert seen == [run.resolve()]


# --------------------------------------------------------------------------- best FID labelling


def _summary(best_metric, best_fid):
    return {"run_id": "c0ffee00c0ffee00", "status": "completed", "step": 10,
            "best": {"metric": best_metric, "value": 0.25, "step": 10}, "best_fid": best_fid}


def test_report_labels_best_fid_only_for_an_fid_primary_metric(isolated, capsys):
    # "fidelity" contains "fid" but summary.is_fid_metric says it is not an FID.
    not_fid = _run(isolated / "a", "a" * 16, summary=_summary("feature_fidelity", None))
    fid = _run(isolated / "b", "b" * 16, summary=_summary("fid10k_train", 12.5))
    assert cli._run_report([str(not_fid), str(fid)], is_json=True) == 0
    records = json.loads(capsys.readouterr().out)
    assert [r["best_fid"] for r in records] == [None, 12.5]


def test_training_result_labels_a_non_fid_best_as_best(isolated):
    from hypergan.bounded_cli_output import TrainingOutput
    for best_metric, best_fid, label in (("feature_fidelity", None, "Best:"), ("fid10k_train", 0.25, "Best FID:")):
        run = _run(isolated / best_metric, summary=_summary(best_metric, best_fid))
        stdout = io.StringIO()
        output = TrainingOutput(stdout, io.StringIO(), False)
        output.viewer_enabled = False
        output.result({"status": "complete", "steps": 10, "total_steps": 10}, run_dir=run, human=True)
        best_line = next(line for line in stdout.getvalue().splitlines() if "feature_fidelity" in line or "fid10k" in line)
        assert best_line.split()[0:2 if label == "Best FID:" else 1] == label.split()


class _Tty(io.StringIO):
    def isatty(self):
        return True


def test_progress_lines_are_whole_lines_even_on_a_tty():
    from hypergan.bounded_cli_output import TrainingOutput
    output = TrainingOutput(io.StringIO(), _Tty(), False, progress_every=1)
    output.progress({"event": "train", "step": 1, "total_steps": 10, "steps_per_second": 2.0})
    output.progress({"event": "checkpoint", "step": 1, "checkpoint_path": "c"})
    output.progress({"event": "evaluation", "step": 1, "metric_id": "fid", "value": 3.0})
    text = output.stderr.getvalue()
    assert "\r" not in text and "\033" not in text
    assert text.count("\n") == 3


# --------------------------------------------------------------------------- hypergan new naming


@pytest.mark.parametrize("extra", [[], ["-t", "dcgan", "-r", "32", "--device", "cpu"]])
def test_new_uses_one_naming_rule_and_prints_absolute_paths(isolated, capsys, extra):
    assert cli.main(["new", "proj", *extra]) in (0, None)
    printed = Path(capsys.readouterr().out.strip())
    assert printed.is_absolute()
    assert printed == (isolated / "proj" / "model.toml").resolve()
    assert cli.main(["new", "single.toml", *extra]) in (0, None)
    assert Path(capsys.readouterr().out.strip()) == (isolated / "single.toml").resolve()


# --------------------------------------------------------------------------- extends resolution


def test_extends_never_searches_the_working_directory(isolated, monkeypatch):
    (isolated / "base.toml").write_text('name = "base"\n[train]\nsteps = 3\n', encoding="utf-8")
    child_dir = isolated / "elsewhere"
    child_dir.mkdir()
    child = child_dir / "child.toml"
    child.write_text('extends = "base.toml"\n', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="base.toml"):
        resolve_inheritance({"extends": "base.toml"}, config_path=child)
    # Provenance uses the same resolver, so it records no parent either.
    assert collect_source_files(child) == [child.resolve()]


def test_extends_resolution_is_shared_by_loading_and_provenance(isolated):
    ws = isolated / "ws"
    (ws / "bases").mkdir(parents=True)
    (ws / "hypergan.toml").write_text("[workspace]\n", encoding="utf-8")
    parent = ws / "bases" / "parent.toml"
    parent.write_text('[train]\nsteps = 5\n', encoding="utf-8")
    (ws / "configs").mkdir()
    child = ws / "configs" / "child.toml"
    child.write_text('extends = "bases/parent.toml"\n', encoding="utf-8")
    merged = resolve_inheritance({"extends": "bases/parent.toml"}, config_path=child, workspace_root=ws)
    assert merged["train"]["steps"] == 5
    assert collect_source_files(child, ws) == [(ws / "hypergan.toml").resolve(), parent.resolve(), child.resolve()]
    packaged = resolve_inheritance({"extends": "gaussian-grid"}, config_path=child)
    assert packaged["networks"]


# --------------------------------------------------------------------------- [checkpoints] validation


@pytest.mark.parametrize("keep_best", [{"metric": "loss/d_total"}, {"mode": "max"}, {}])
def test_keep_best_count_defaults_to_the_policy_default(keep_best):
    count = DEFAULT_CHECKPOINT_POLICY["keep_best"]["count"]
    assert normalize_checkpoint_policy({"keep_best": keep_best})["keep_best"]["count"] == count == 2
    model = hg.override(hg.load(EXAMPLES / "two_critics.toml"), {"checkpoints": {"keep_best": keep_best}})
    assert hg.lower(model)["checkpoints"]["keep_best"]["count"] == count


@pytest.mark.parametrize("table, message", [
    ({"export_bundle": "sometimes"}, "checkpoints.export_bundle must be one of"),
    ({"keep_best": {"metric": "x", "cuont": 1}}, "checkpoints.keep_best has unknown field"),
    ({"keep_last": -1}, "checkpoints.keep_last must be a non-negative integer"),
    ({"keep_everything": 1}, "Unknown checkpoints field"),
])
def test_lowering_validates_checkpoints_like_the_policy(table, message):
    model = hg.load(EXAMPLES / "two_critics.toml")
    spec = hg.override(model, {"checkpoints": table}).spec
    with pytest.raises(ModelFileError, match=message):
        lower(spec, EXAMPLES)


# --------------------------------------------------------------------------- hidden magic


def test_clearing_a_loss_id_is_explicit_and_never_written():
    model = hg.load(EXAMPLES / "two_critics.toml")
    cleared = hg.override(model, {"losses.reconstruction.id": None})
    assert all("__id_required__" not in loss for loss in cleared.spec["losses"])
    assert "__id_required__" not in hg.dumps(cleared)
    with pytest.raises(ModelFileError, match="id is required"):
        hg.lower(cleared)


def test_load_of_a_pt_bundle_points_to_load_bundle(isolated):
    bundle = isolated / "model.pt"
    bundle.write_bytes(b"not read")
    with pytest.raises(ModelFileError, match="hg.load_bundle"):
        hg.load(bundle)
    with pytest.raises(ModelFileError, match="hg.load_bundle"):
        hg.override(bundle, {"train.steps": 1})


# --------------------------------------------------------------------------- API surface


def test_aliases_resolve_but_only_canonical_names_are_exported():
    for alias, canonical in hg.ALIASES.items():
        assert alias not in hg.__all__ and alias not in hypergan.__all__
        assert canonical in hg.__all__
        assert callable(getattr(hypergan, alias))
    assert hypergan.prune is hg.gc and hypergan.runs is hg.list_runs
    assert hypergan.validate_config is hg.validate and hypergan.checkpoint_status is hg.request_status


def test_ps_and_gc_do_not_shadow_all():
    assert "all" not in inspect.signature(hg.ps).parameters
    assert "all" not in inspect.signature(hg.gc).parameters
    assert "all_runs" in inspect.signature(hg.ps).parameters


def test_queue_run_forwards_no_web(monkeypatch):
    import hypergan.queue as queue
    seen = {}
    monkeypatch.setattr(queue, "queue_run", lambda **kwargs: seen.update(kwargs))
    hg.queue_run(once=True, no_web=True)
    assert seen["no_web"] is True and seen["once"] is True


def test_ensure_dashboard_running_forwards_host_port_and_auth(monkeypatch, tmp_path):
    import hypergan.web_autostart as web_autostart
    calls = []
    monkeypatch.setattr(web_autostart, "ensure_dashboard_running", lambda root, **kwargs: calls.append((root, kwargs)) or "url")
    hg.ensure_dashboard_running(tmp_path, host="127.0.0.1", port=9001, auth="token")
    assert calls[0] == (tmp_path, {"check_version": True, "run_id": None, "host": "127.0.0.1", "port": 9001, "auth": "token"})
    hg.ensure_dashboard_running(tmp_path)
    assert calls[1][1] == {"check_version": True, "run_id": None}


def test_new_project_passes_name_through(isolated, monkeypatch):
    seen = []
    monkeypatch.setattr(hg, "init_project", lambda project, **kwargs: seen.append(kwargs) or {})
    hg.new(project=isolated / "p", name="faces", template="gaussian-grid")
    assert seen[0]["output_config"] == Path("configs") / "faces.toml"
    with pytest.raises(TypeError, match="name or output_path"):
        hg.new(project=isolated / "p", name="faces", output_path="configs/other.toml")


def test_data_check_does_not_trust_the_inspection_cache(isolated, monkeypatch):
    import hypergan.data as data
    seen = {}

    class FakeFolder:
        inventory = {}

        def __init__(self, **kwargs):
            seen.update(kwargs)

        def resume_identity(self):
            return {}

    import hypergan.config as config
    monkeypatch.setattr(data, "ImageFolder", FakeFolder)
    monkeypatch.setattr(config, "load_config", lambda path: {"data": {"factory": "image_folder", "args": {"root": str(isolated)}}})
    hg.data_check(isolated / "model.toml")
    assert seen["manifest_cache"] is False and seen["root"] == str(isolated)


# --------------------------------------------------------------------------- recipe generator


def test_one_resolution_preset_rule():
    from hypergan.recipe_generator import preset_resolution
    assert [preset_resolution(w, h) for w, h in ((8, 8), (32, 20), (33, 33), (64, 100), (128, 128), (129, 50), (999, 9))] == \
        [32, 32, 64, 128, 128, 256, 256]


def test_recipe_comments_take_the_data_table():
    from hypergan.recipe_generator import _add_recipe_comments
    body = '[data]\nbuiltin = "image_folder"\n'
    text = _add_recipe_comments(body, "m.toml", template="dcgan",
                                data_spec={"builtin": "image_folder", "args": {"root": "/imgs", "height": 32, "width": 32}})
    assert 'root = "/imgs", height = 32, width = 32' in text
    with pytest.raises(TypeError):
        _add_recipe_comments(body, "m.toml", template="dcgan")


# --------------------------------------------------------------------------- load_inference


def _checkpointed_run(directory):
    torch = pytest.importorskip("torch")
    from hypergan.config import resolve_config
    from hypergan.recipes import ComponentGraph, make_prior
    run = Path(directory)
    ckpt_dir = run / "checkpoints" / "att-1-step-00000100-uuid1"
    ckpt_dir.mkdir(parents=True)
    cfg = resolve_config({"components": {
        "generator": {"factory": "linear", "args": {"in_features": 2, "out_features": 2}, "inputs": {"z": "latent"}},
        "discriminator": {"factory": "linear", "args": {"in_features": 2, "out_features": 1}, "inputs": {"x": "candidate"}},
    }})
    graph = ComponentGraph(cfg["components"])
    torch.save({"step": 100,
                "ema_graph": {f"models.generator.{k}": v for k, v in graph.models["generator"].state_dict().items()},
                "ema_prior": make_prior(cfg["prior"], device="cpu").state_dict(),
                "buffers": {"ema_graph": {}, "ema_prior": {}}, "last_batch": {}}, ckpt_dir / "state.pt")
    (ckpt_dir / "manifest.json").write_text(json.dumps({"step": 100, "config": cfg}), encoding="utf-8")
    return run, ckpt_dir


def test_load_inference_reads_a_checkpoint_without_writing(tmp_path):
    run, _ = _checkpointed_run(tmp_path / "run")
    before = sorted(p.relative_to(run) for p in run.rglob("*"))
    state, _, graph, prior, sha = hg.load_inference(run)
    assert state["kind"] == "ema-inference" and state["step"] == 100
    assert graph is not None and prior is not None and len(sha) == 64
    assert sorted(p.relative_to(run) for p in run.rglob("*")) == before
    assert not (run / "model.pt").exists()


def test_load_inference_reports_checkpoint_errors(tmp_path):
    run, ckpt_dir = _checkpointed_run(tmp_path / "run")
    (ckpt_dir / "state.pt").write_bytes(b"corrupt")
    with pytest.raises(Exception) as exc:
        hg.load_inference(run)
    assert not isinstance(exc.value, FileNotFoundError)  # the real error, not "no bundle"
    assert not (run / "model.pt").exists()
    empty = tmp_path / "empty"
    (empty / "checkpoints").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No inference bundle or checkpoint"):
        hg.load_inference(empty)
