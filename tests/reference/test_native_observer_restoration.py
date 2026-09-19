"""Numerical restoration errors must not become optional delivery errors."""
import json

import pytest

from hypergan.checkpoints import capture_rng, restore_rng
from hypergan.config import write_default
from hypergan.training import train


def test_native_rng_restore_failure_stops_without_saving_changed_state(tmp_path, monkeypatch):
    import hypergan.single_execution as native
    saved_rng = capture_rng()
    fail_next = False
    calls = []
    def callback(event):
        nonlocal fail_next
        calls.append(event['event'])
        if event['event'] == 'train':
            fail_next = True
    def broken_restore(state):
        nonlocal fail_next
        if fail_next:
            fail_next = False
            raise RuntimeError('injected native RNG restoration failure')
        return restore_rng(state)
    monkeypatch.setattr(native, 'restore_rng', broken_restore)
    config = write_default(tmp_path / 'project', device="cpu")
    try:
        with pytest.raises(RuntimeError, match='native RNG restoration failure'):
            train(config, tmp_path / 'run', steps=3, on_event=callback)
        manifest = json.loads((tmp_path / 'run/manifest.json').read_text())
        events = [json.loads(line) for line in (tmp_path / 'run/events.jsonl').read_text().splitlines()]
        assert manifest['status'] == 'failed'
        assert manifest['steps'] == 1 and manifest['last_durable_step'] == 0
        assert manifest['possible_lost_steps'] == 1
        assert not manifest['observation_errors']
        assert calls.count('train') == 1 and calls[-1] == 'failed'
        assert [event['step'] for event in events if event['event'] == 'checkpoint'] == [0]
        assert not manifest.get('bundle_path') and not manifest.get('sample_path')
    finally:
        restore_rng(saved_rng)
