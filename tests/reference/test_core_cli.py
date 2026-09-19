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
    config = write_default(tmp_path / "project")
    run = tmp_path / "run"
    first = cli(tmp_path, "train", config, "--run-dir", run, "--stop-after-steps", 2,
                "--checkpoint-every", 1, "--progress-json")
    assert first.returncode == 0, first.stderr
    rows = [json.loads(line) for line in first.stdout.splitlines()]
    assert [row["step"] for row in rows if row["event"] == "train"] == [1, 2]
    before = rows[-1]["manifest"]
    assert before["status"] != "complete" and before["last_durable_step"] == 2
    old_sample = Path(before["sample_path"])
    saved = old_sample.read_bytes()
    second = cli(tmp_path, "resume", run)
    assert second.returncode == 0, second.stderr
    after = json.loads(second.stdout)
    assert after["status"] == "complete" and after["steps"] == 5
    assert after["attempt_id"] != before["attempt_id"]
    assert after["sample_path"] != before["sample_path"]
    assert old_sample.read_bytes() == saved
    assert "step 3:" in second.stderr
    result = cli(tmp_path, "sample", run, "--count", 3)
    assert result.returncode == 0, result.stderr
    assert json.loads(Path(result.stdout.strip()).read_text())["step"] == 5


def test_sampling_preserves_global_rng_and_existing_outputs(tmp_path):
    config = write_default(tmp_path / "project")
    run = tmp_path / "run"
    result = cli(tmp_path, "train", config, "--run-dir", run)
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


def test_progress_is_observable_before_process_completion(tmp_path):
    config = write_default(tmp_path / "project")
    run = tmp_path / "run"
    # Stop reading once the first update arrives, before the bounded long attempt
    # can finish. communicate then drains both pipes and enforces a timeout.
    process = subprocess.Popen(
        [sys.executable, "-I", "-m", "hypergan", "train", str(config), "--run-dir", str(run),
         "--steps", "1000", "--stop-after-steps", "100", "--progress-json"],
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
                pytest.fail("No live training event was emitted")
            if json.loads(line).get("event") == "train":
                assert process.poll() is None
                break
        process.wait(timeout=45)
        reader.join(timeout=5)
        stderr = process.stderr.read()
        assert process.returncode == 0, stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
