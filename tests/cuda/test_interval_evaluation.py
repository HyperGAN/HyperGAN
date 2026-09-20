"""Opt-in native CUDA interval evaluation and exact current-run recovery gate."""
from pathlib import Path
import runpy

import torch

from hypergan.training import train, resume
from hypergan.checkpoints import read_checkpoint
from hypergan.distributed_checkpoints import _digest


def test_cuda_interval_evaluation_preserves_state_and_snapshot_recovery(tmp_path, monkeypatch):
    assert torch.cuda.is_available(), 'Interval evaluation CUDA acceptance requires a local GPU'
    helper = runpy.run_path(str(Path(__file__).parents[1] / 'reference/test_interval_evaluation.py'))
    path = helper['config'](tmp_path, monkeypatch, interval=2)
    path.write_text(path.read_text().replace('device = "cpu"', 'device = "cuda"') +
                    '\n[training.backend]\ndeterministic_algorithms = true\n')
    plain = tmp_path / 'plain.toml'
    plain.write_text(path.read_text().replace('trigger = "interval"\nevery_steps = 2\non_busy = "skip"', 'trigger = "manual"'))
    baseline = train(plain, tmp_path / 'baseline', checkpoint_every=1)
    full = train(path, tmp_path / 'full', checkpoint_every=1)
    stopped = train(path, tmp_path / 'recovery', checkpoint_every=1, stop_after_steps=2)
    resumed = resume(tmp_path / 'recovery')
    recovered = resume(tmp_path / 'recovery', checkpoint=stopped['checkpoint_path'])
    expected = _digest(read_checkpoint(tmp_path / 'baseline', baseline['checkpoint_path'])[2])
    for root, manifest in [(tmp_path / 'full', full), (tmp_path / 'recovery', resumed),
                           (tmp_path / 'recovery', recovered)]:
        assert _digest(read_checkpoint(root, manifest['checkpoint_path'])[2]) == expected
    results = helper['receipts'](tmp_path / 'recovery')
    assert sorted(item['result']['step'] for item in results) == [2, 4, 4]
    assert all(item['status'] == 'complete' for item in results)
    assert all(item['result']['protocol']['runtime']['device'].startswith('cuda') for item in results)
    step_four = [item['result']['value'] for item in results if item['result']['step'] == 4]
    assert step_four[0] == step_four[1]
