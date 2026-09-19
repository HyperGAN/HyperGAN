"""Run observation and checkpoint requests work in a base-only installation."""
import json
import subprocess
import sys


def cli(tmp_path, *args):
    return subprocess.run([sys.executable, "-I", "-m", "hypergan", *map(str, args)],
                          cwd=tmp_path, text=True, capture_output=True, timeout=30)


def test_event_pages_and_checkpoint_receipts_without_training(tmp_path):
    manifest = {"schema_version": 1, "run_id": "run-one", "attempt_id": "attempt-one", "status": "running"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [dict(schema_version=1, run_id="run-one", attempt_id="attempt-one", sequence=i,
                 step=i, event="train") for i in (1, 2)]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    first = cli(tmp_path, "events", tmp_path, "--limit", 1)
    assert first.returncode == 0, first.stderr
    page = json.loads(first.stdout)
    assert page["events"] == rows[:1] and page["has_more"]
    second = cli(tmp_path, "events", tmp_path, "--cursor", page["cursor"])
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)["events"] == rows[1:]
    request = cli(tmp_path, "checkpoint", tmp_path, "--request-id", "save-one")
    assert request.returncode == 0, request.stderr
    receipt = json.loads(request.stdout)
    assert receipt["status"] == "pending"
    assert receipt["request"]["attempt_id"] == "attempt-one"
    retry = cli(tmp_path, "checkpoint", tmp_path, "--request-id", "save-one")
    assert retry.returncode == 0, retry.stderr
    assert json.loads(retry.stdout) == receipt
    status = cli(tmp_path, "checkpoint", tmp_path, "--status", "save-one")
    assert status.returncode == 0, status.stderr
    assert json.loads(status.stdout) == receipt
    manifest["status"] = "complete"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    retry = cli(tmp_path, "checkpoint", tmp_path, "--request-id", "save-one")
    assert retry.returncode == 0, retry.stderr
    new = cli(tmp_path, "checkpoint", tmp_path, "--request-id", "save-two")
    assert new.returncode != 0 and "not running" in new.stderr
    probe = subprocess.run([sys.executable, "-I", "-c",
        "import sys, hypergan.run_events, hypergan.run_requests; "
        "assert not {'torch', 'particlegan', 'numpy', 'PIL'} & sys.modules.keys()"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert probe.returncode == 0, probe.stderr


def test_observer_usage_errors_are_actionable(tmp_path):
    for args in [("events", tmp_path, "--limit", "0"),
                 ("checkpoint", tmp_path, "--status", "missing", "--attempt-id", "old"),
                 ("train", "project", "--run-dir", "run", "--preview-every", "1", "--no-previews"),
                 ("resume", "run", "--preview-keep", "0")]:
        result = cli(tmp_path, *args)
        assert result.returncode != 0
        assert "error:" in result.stderr and "Traceback" not in result.stderr
