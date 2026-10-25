"""Regressions for the 2026-10-25 review of the CLI and Python API layer."""

import inspect
import io
import json
from pathlib import Path
import sys

import pytest

import hypergan.api as hg
from hypergan import cli


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERGAN_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("HYPERGAN_STATE_DIR", raising=False)
    monkeypatch.delenv("HYPERGAN_WORKSPACE", raising=False)
    monkeypatch.delenv("HYPERGAN_DEBUG", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _model_file(directory, name="a"):
    return hg.save(hg.override(hg.packaged("gaussian-grid"), {"train.device": "cpu"}),
                   Path(directory) / f"{name}.toml")


def _run_dir(directory, config=None, events=()):
    run = Path(directory)
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"status": "stopped", "config": config or {}}))
    (run / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    return run


# --------------------------------------------------------------------------- hg.new / README


def test_new_project_accepts_documented_dataset_keyword(isolated):
    from PIL import Image
    images = isolated / "images"
    images.mkdir()
    for index in range(2):
        Image.new("RGB", (32, 32), (index * 80, 10, 10)).save(images / f"{index}.png")

    info = hg.new(project=isolated / "p", template="dcgan", dataset=images, device="cpu")

    assert info["template"] == "dcgan"
    assert Path(info["recipe_path"]).name == "dcgan.toml"


def test_new_rejects_unknown_keywords(isolated):
    with pytest.raises(TypeError):
        hg.new(project=isolated / "p", datset="./images")
    with pytest.raises(TypeError):
        hg.new(isolated / "a.toml", template="gaussian-grid", bogus=1)
    with pytest.raises(TypeError, match="dataset"):
        hg.new(project=isolated / "p", data_dir="a", dataset="b")


def test_cli_new_project_passes_the_template_through(isolated, monkeypatch):
    seen = []

    def fake_init_project(project_dir, **kwargs):
        seen.append(kwargs["template"])
        return {"project_dir": str(project_dir), "workspace_config": "w", "recipe_path": "r",
                "template": kwargs["template"], "next_steps": []}

    monkeypatch.setattr(hg, "init_project", fake_init_project)
    assert cli.main(["new", "--project", "p1"]) == 0
    assert cli.main(["new", "--project", "p2", "-t", "dcgan"]) == 0
    # "No template" stays distinguishable from an explicit dcgan.
    assert seen == [None, "dcgan"]


def test_documented_api_example_has_a_main_guard():
    assert 'if __name__ == "__main__":' in hg.__doc__
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")
    api_block = readme.split("```python\nimport hypergan as hg\n", 1)[1].split("```", 1)[0]
    assert 'if __name__ == "__main__":' in api_block
    assert "dataset=" not in api_block


# --------------------------------------------------------------------------- run directories


def test_default_run_dir_is_a_fresh_slug_like_the_cli(isolated):
    model = hg.load(_model_file(isolated))
    first = hg._default_run_dir(model)
    assert first.parent == isolated.resolve()
    assert first.name.startswith("a-") and first.name != "a"
    first.mkdir()
    assert hg._default_run_dir(model) != first


def test_default_run_dir_for_packaged_model_is_not_nested(isolated):
    run_dir = hg._default_run_dir(hg.packaged("gaussian-grid"))
    assert run_dir.parent == isolated.resolve()
    assert run_dir.name.startswith("gaussian-grid-")


def test_default_run_dir_explicit_name_and_workspace(isolated):
    workspace = isolated / "ws"
    (workspace / "configs").mkdir(parents=True)
    (workspace / "hypergan.toml").write_text('[workspace]\nruns_root = "out"\n')
    model = hg.load(_model_file(workspace / "configs", "b"))
    assert hg._default_run_dir(model).parent == (workspace / "out").resolve()
    assert hg._default_run_dir(model, name="exp/one") == (workspace / "out" / "exp-one").resolve()


def test_cli_and_api_share_the_default_run_dir(isolated, monkeypatch):
    seen = []
    real = hg._default_run_dir

    def spy(model, name=None):
        result = real(model, name=name)
        seen.append(result)
        return result

    class Stop(Exception):
        pass

    def stop_prepare(*args, **kwargs):
        raise Stop

    monkeypatch.setattr(hg, "_default_run_dir", spy)
    monkeypatch.setattr(hg, "prepare", stop_prepare)
    config = _model_file(isolated)
    with pytest.raises(Stop):
        cli.main(["train", str(config), "--no-server"])
    assert len(seen) == 1 and seen[0].name.startswith("a-")


# --------------------------------------------------------------------------- resume re-exec


def test_resume_reexec_keeps_module_invocation(isolated, monkeypatch):
    run = _run_dir(isolated / "run", config={"run": {"env": {"HG_REVIEW_REEXEC_PROBE": "1"}}})
    monkeypatch.delenv("HG_REVIEW_REEXEC_PROBE", raising=False)
    monkeypatch.delenv("_HYPERGAN_REEXEC", raising=False)
    # `python -m hypergan resume run`: argv[0] is the package's __main__.py, with no "-m".
    monkeypatch.setattr(sys, "argv", [str(Path(cli.__file__).with_name("__main__.py")), "resume", str(run)])
    calls = []

    class Exec(BaseException):
        pass

    def fake_execve(executable, argv, env):
        calls.append((executable, argv, env))
        raise Exec

    monkeypatch.setattr(cli.os, "execve", fake_execve)
    with pytest.raises(Exec):
        cli.main(["resume", str(run), "--no-server"])
    executable, argv, env = calls[0]
    assert argv == [executable, "-m", "hypergan", "resume", str(run), "--no-server"]
    assert env["HG_REVIEW_REEXEC_PROBE"] == "1" and env["_HYPERGAN_REEXEC"] == "1"


# --------------------------------------------------------------------------- logs


def test_logs_raw_follow_does_not_raise_and_stops_at_terminal_event(isolated, capsys):
    events = [{"event": "train", "step": 1}, {"event": "complete", "step": 2}]
    run = _run_dir(isolated / "run", events=events)

    stream = io.StringIO()
    lines = hg.logs(run, raw=True, follow=True, stream=stream)
    assert [json.loads(line)["event"] for line in lines] == ["train", "complete"]
    assert stream.getvalue().splitlines() == lines

    assert hg.logs(run, raw=True, follow=True) == lines
    assert capsys.readouterr().out.splitlines() == lines
    assert hg.logs(run, raw=True) == lines
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- doctor / sample / queue / --set


def test_doctor_reports_the_registry_directory(isolated, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setenv("HYPERGAN_HOME", str(isolated / "home"))
    monkeypatch.setenv("HYPERGAN_STATE_DIR", str(isolated / "state-dir"))
    assert cli.main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["workspace"]["state_dir"] == str((isolated / "home").resolve())


def test_sample_sampler_writes_its_own_file(isolated, monkeypatch, capsys):
    run = _run_dir(isolated / "run")
    grid = isolated / "grid.png"

    def fake_sample(run_dir, *, output=None, **kwargs):
        Path(output).write_bytes(b"grid")
        return hg.Sample(step=1, name="sample", data=None, inputs={}, particle_ids=None,
                         source="inference", path=Path(output))

    class View:
        def save(self, path):
            Path(path).write_bytes(b"view")
            return Path(path)

    monkeypatch.setattr(hg, "sample", fake_sample)
    monkeypatch.setattr(hg, "view", lambda item, sampler, run=None: View())
    assert cli.main(["sample", str(run), "--seed", "1", "--output", str(grid),
                     "--sampler", "mod:fn", "--json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert grid.read_bytes() == b"grid"
    view_path = Path(printed["view"])
    assert view_path != grid and view_path.read_bytes() == b"view"


def test_queue_install_service_does_not_advertise_a_fixed_user_flag():
    parser = cli._parser()
    queue = parser._subparsers._group_actions[0].choices["queue"]
    install = queue._subparsers._group_actions[0].choices["install-service"]
    assert "--user" not in install.format_help()
    # Still accepted, so existing scripts keep working; units are always user units.
    args = parser.parse_args(["queue", "install-service", "--user", "--dry-run"])
    assert args.queue_command == "install-service"


def test_set_value_none_is_a_string_and_null_unsets():
    assert cli._parse_set_value("none") == "none"
    assert cli._parse_set_value("None") == "None"
    assert cli._parse_set_value("null") is None


# --------------------------------------------------------------------------- help / errors


def test_top_level_help_lists_every_subcommand():
    parser = cli._parser()
    names = parser._subparsers._group_actions[0].choices
    text = parser.format_help()
    missing = [name for name in names if name not in text]
    assert not missing


def test_debug_shows_the_traceback(isolated, monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise KeyError("boom")

    monkeypatch.setattr(hg, "status", broken)
    assert cli.main(["status"]) == 1
    quiet = capsys.readouterr().err
    assert "Traceback" not in quiet and "error: 'boom'" in quiet

    assert cli.main(["--debug", "status"]) == 1
    assert "Traceback" in capsys.readouterr().err

    monkeypatch.setenv("HYPERGAN_DEBUG", "1")
    assert cli.main(["status"]) == 1
    assert "Traceback" in capsys.readouterr().err


# --------------------------------------------------------------------------- one diff, aligned defaults


def test_cli_diff_uses_the_api_diff(isolated, monkeypatch, capsys):
    config = _model_file(isolated)
    api_result = hg.diff(str(config), sets={"optimizer.lr": 0.0123})
    assert any(value["b"] == 0.0123 for value in api_result["differences"].values())

    assert cli.main(["diff", str(config), "-s", "optimizer.lr=0.0123", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == api_result
    assert not hasattr(cli, "_extract_diffable_config")


def test_api_alias_defaults_match_the_cli():
    default = lambda fn, name: inspect.signature(fn).parameters[name].default
    assert default(hg.runs_ls, "sort") == default(hg.list_runs, "sort") == "updated"
    assert default(hg.runs_compare, "metric") == default(hg.compare, "metric") == "loss/g_total"
    parser = cli._parser()
    compare = parser._subparsers._group_actions[0].choices["compare"]
    assert compare.get_default("metric") == "loss/g_total"


# --------------------------------------------------------------------------- argument checks


@pytest.mark.parametrize("argv", [
    ["logs", "-n", "0"],
    ["logs", "-n", "-3"],
    ["stop", "--timeout", "-1"],
    ["stop", "--timeout", "nan"],
    ["queue", "run", "--poll-interval", "0"],
    ["queue", "status", "--lines", "0"],
    ["download-data", "--timeout", "-5"],
    ["qualify", "--timeout", "0.5"],
    ["qualify", "--timeout", "nan"],
])
def test_numeric_options_reject_nonsense(argv):
    with pytest.raises(SystemExit) as exit_info:
        cli._parser().parse_args(argv)
    assert exit_info.value.code == 2


@pytest.mark.parametrize("command", ["validate", "explain"])
def test_project_commands_refuse_to_pick_among_several_recipes(isolated, command, capsys):
    project = isolated / "proj"
    (project / "configs").mkdir(parents=True)
    _model_file(project / "configs", "one")
    _model_file(project / "configs", "two")
    assert cli.main([command, str(project)]) == 1
    assert "Multiple recipes found" in capsys.readouterr().err
