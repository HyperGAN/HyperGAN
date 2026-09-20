"""Installed-package recovery workflow and isolated inference publication."""
import json
from pathlib import Path
import random
import queue
import subprocess
import sys
import threading

import numpy as np
import pytest
import torch

from hypergan.artifacts import sample, save_bundle
from hypergan.config import resolve_config, write_default


def cli(tmp_path, *args):
    return subprocess.run([sys.executable, "-I", "-m", "hypergan", *map(str, args)],
                          cwd=tmp_path, capture_output=True, text=True, timeout=45)


def test_cli_stop_resume_and_json_progress(tmp_path):
    config = write_default(tmp_path / "project", device="cpu")
    run = tmp_path / "run"
    first = cli(tmp_path, "train", config, "--run-dir", run, "--no-server", "--stop-after-steps", 2,
                "--checkpoint-every", 1, "--preview-every", 1, "--preview-keep", 2, "--progress-json", "--progress-every", 1)
    assert first.returncode == 0, first.stderr
    rows = [json.loads(line) for line in first.stdout.splitlines()]
    assert [row["step"] for row in rows if row["event"] == "train"] == [1, 2]
    before = rows[-1]["manifest"]
    assert before["status"] != "complete" and before["last_durable_step"] == 2
    previews = json.loads((run / "previews" / "index.json").read_text())
    first_steps = [record["step"] for record in previews["previews"]]
    assert first_steps == sorted(set(first_steps)) and first_steps[0] == 1
    assert set(first_steps) <= {1, 2}
    assert len(first_steps) + before.get("skipped_previews_busy", 0) == 2
    assert not before["observation_errors"]
    assert all(record["identity"]["attempt_id"] == before["attempt_id"] for record in previews["previews"])
    old_sample = Path(before["sample_path"])
    saved = old_sample.read_bytes()
    second = cli(tmp_path, "resume", run, "--no-server")
    assert second.returncode == 0, second.stderr
    after = json.loads(second.stdout)
    assert after["status"] == "complete" and after["steps"] == 5
    previews = json.loads((run / "previews" / "index.json").read_text())
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    measured = [row for row in events if row["event"] == "preview"]
    resumed_steps = [row["step"] for row in measured if row["attempt_id"] == after["attempt_id"]]
    assert resumed_steps == sorted(set(resumed_steps)) and resumed_steps[0] == 3
    assert set(resumed_steps) <= {3, 4, 5}
    skipped = after.get("skipped_previews_busy", 0) - before.get("skipped_previews_busy", 0)
    assert len(resumed_steps) + skipped == 3
    assert not after["observation_errors"]
    assert [record["step"] for record in previews["previews"]] == (first_steps + resumed_steps)[-2:]
    for record in previews["previews"]:
        payload = json.loads(Path(record["path"]).read_text())
        assert payload["step"] == record["step"]
        assert payload["identity"] == record["identity"]
    assert after["preview_every"] == 1 and after["preview_keep"] == 2
    assert after["attempt_id"] != before["attempt_id"]
    assert after["sample_path"] != before["sample_path"]
    assert old_sample.read_bytes() == saved
    assert "step 3:" in second.stderr
    result = cli(tmp_path, "sample", run, "--count", 3)
    assert result.returncode == 0, result.stderr
    assert json.loads(Path(result.stdout.strip()).read_text())["step"] == 5


def test_sampling_preserves_global_rng_and_existing_outputs(tmp_path):
    config = write_default(tmp_path / "project", device="cpu")
    run = tmp_path / "run"
    result = cli(tmp_path, "train", config, "--run-dir", run, "--no-server")
    assert result.returncode == 0, result.stderr
    torch_state = torch.get_rng_state().clone()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    first = sample(run, count=3)
    second = sample(run, count=3)
    assert first != second
    assert first.read_bytes() == second.read_bytes()
    assert torch.equal(torch_state, torch.get_rng_state())
    assert python_state == random.getstate()
    current = np.random.get_state()
    assert numpy_state[0] == current[0] and np.array_equal(numpy_state[1], current[1])
    assert numpy_state[2:] == current[2:]
    with pytest.raises(FileExistsError):
        sample(run, count=3, output=first)
    assert first.read_bytes() == second.read_bytes()
    assert not list(run.glob(".sample-*.tmp"))


class BufferedGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.register_buffer("offset", torch.zeros(2), persistent=False)

    def forward(self, x):
        return self.linear(x) + self.offset


def test_inference_restores_nonpersistent_registered_buffers(tmp_path):
    from hypergan.training import ReferenceTrainer
    config = resolve_config({})
    config["components"]["generator"].update(factory=f"{__name__}:BufferedGenerator", args={})
    trainer = ReferenceTrainer(config)
    trainer.ema_graph.models["generator"].offset.copy_(torch.tensor([5., 7.]))
    save_bundle(tmp_path, trainer, {"real": torch.zeros(16, 2)})
    with torch.inference_mode():
        z, _ = trainer.ema_prior.sample(3, generator=torch.Generator().manual_seed(42))
        expected = trainer.ema_graph.generate(z, {})["generated"]
    output = sample(tmp_path, count=3, seed=42)
    actual = torch.tensor(json.loads(output.read_text())["samples"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_live_progress_and_cross_process_checkpoint_request(tmp_path):
    config = write_default(tmp_path / "project", device="cpu")
    run = tmp_path / "run"
    # Submit from another process as soon as the start event arrives, before the
    # bounded attempt finishes. The reader keeps draining progress throughout.
    process = subprocess.Popen(
        [sys.executable, "-I", "-m", "hypergan", "train", str(config), "--run-dir", str(run),
         "--steps", "1000", "--stop-after-steps", "100", "--progress-json", "--no-server"],
        cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        lines = queue.Queue()
        def read_progress():
            for line in process.stdout:
                lines.put(line)
            lines.put(None)
        reader = threading.Thread(target=read_progress, daemon=True)
        reader.start()
        while True:
            line = lines.get(timeout=45)
            if line is None:
                pytest.fail("No live start event was emitted")
            if json.loads(line).get("event") == "start":
                assert process.poll() is None
                requested = cli(tmp_path, "checkpoint", run, "--request-id", "live-save")
                assert requested.returncode == 0, requested.stderr
                break
        process.wait(timeout=45)
        reader.join(timeout=5)
        stderr = process.stderr.read()
        assert process.returncode == 0, stderr
        receipt = cli(tmp_path, "checkpoint", run, "--status", "live-save")
        assert receipt.returncode == 0, receipt.stderr
        acknowledged = json.loads(receipt.stdout)
        assert acknowledged["status"] == "succeeded"
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


def test_cli_viewer_preserves_numerics_and_machine_output(tmp_path):
    """Optional web qualification belongs to the explicitly provisioned web+train job."""
    import importlib.util
    if any(importlib.util.find_spec(name) is None for name in ('starlette', 'uvicorn', 'wasmtime')):
        # Base numerical CI has no web dependencies: prove explicit headless CLI.
        config = write_default(tmp_path / 'project', device='cpu')
        result = cli(tmp_path, 'train', config, '--run-dir', tmp_path / 'headless', '--no-server')
        assert result.returncode == 0, result.stderr
        assert 'Viewer:' not in result.stderr
        assert json.loads(result.stdout)['status'] == 'complete'
        return
    config = write_default(tmp_path / 'project', device='cpu')
    headless = cli(tmp_path, 'train', config, '--run-dir', tmp_path / 'headless', '--no-server')
    assert headless.returncode == 0, headless.stderr
    viewed = tmp_path / 'viewed'
    try:
        first = cli(tmp_path, 'train', config, '--run-dir', tmp_path / 'viewed', '--server',
                    '--stop-after-steps', 2, '--progress-json')
        assert first.returncode == 0, first.stderr
        assert 'Viewer:' in first.stderr
        rows = [json.loads(line) for line in first.stdout.splitlines()]
        assert rows[-1]['manifest']['steps'] == 2
        initial_status = cli(tmp_path, 'server-status', viewed)
        assert initial_status.returncode == 0, initial_status.stderr
        initial_viewer = json.loads(initial_status.stdout)
        assert initial_viewer['status'] == 'ready'
        resumed = cli(tmp_path, 'resume', viewed, '--server')
        assert resumed.returncode == 0, resumed.stderr
        expected, actual = json.loads(headless.stdout), json.loads(resumed.stdout)
        assert expected['steps'] == actual['steps'] == 5
        expected_sample = json.loads(Path(expected['sample_path']).read_text())
        actual_sample = json.loads(Path(actual['sample_path']).read_text())
        for sample_record in (expected_sample, actual_sample):
            sample_record.pop('identity')
            sample_record.pop('bundle_sha256')
        assert expected_sample == actual_sample
        def same_state(left, right):
            if isinstance(left, torch.Tensor):
                assert torch.equal(left, right)
            elif isinstance(left, dict):
                assert left.keys() == right.keys()
                for key in left:
                    same_state(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                assert len(left) == len(right)
                for a, b in zip(left, right):
                    same_state(a, b)
            else:
                assert left == right
        same_state(torch.load(Path(expected['checkpoint_path']) / 'state.pt', weights_only=True),
                   torch.load(Path(actual['checkpoint_path']) / 'state.pt', weights_only=True))
        for root in [tmp_path / 'headless', tmp_path / 'viewed']:
            events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
            losses = [{key: value for key, value in event['metrics'].items() if not key.startswith('timing/')}
                      for event in events if event['event'] == 'train']
            if root.name == 'headless':
                expected_losses = losses
            else:
                assert losses == expected_losses
        current_status = cli(tmp_path, 'server-status', viewed)
        assert current_status.returncode == 0, current_status.stderr
        current_viewer = json.loads(current_status.stdout)
        assert current_viewer['status'] == 'ready'
        assert current_viewer['server_instance_id'] == initial_viewer['server_instance_id']
        assert current_viewer['supervisor_pid'] == initial_viewer['supervisor_pid']
        assert current_viewer['processes'] == initial_viewer['processes']
        receipts = list((viewed / 'observations').glob('viewer-*.json'))
        assert len(receipts) == 1
        assert json.loads(receipts[0].read_text())['status'] == 'ready'
        stopped = cli(tmp_path, 'stop-server', viewed)
        assert stopped.returncode == 0, stopped.stderr
        assert json.loads(stopped.stdout)['status'] == 'stopped'
        assert json.loads(receipts[0].read_text())['status'] == 'stopped'
        assert not Path(current_viewer['session_file']).exists()
    finally:
        # Test failures must not leave the now-persistent viewer or projector.
        cli(tmp_path, 'stop-server', viewed)
