"""Extracted lifecycle source is part of strict full-checkpoint identity."""
import hashlib
import importlib
from pathlib import Path

import pytest

from hypergan.config import write_default


@pytest.mark.parametrize('name', ['hypergan.run_controller', 'hypergan.single_execution'])
def test_extracted_source_bytes_are_strict_recovery_identity(tmp_path, monkeypatch, name):
    from hypergan.checkpoints import read_checkpoint
    from hypergan.training import resume, train
    module = importlib.import_module(name)
    config = write_default(tmp_path / 'config')
    root = tmp_path / 'run'
    train(config, root, stop_after_steps=1)
    _, metadata, _ = read_checkpoint(root)
    source_bytes = Path(module.__file__).read_bytes()
    assert metadata['implementation'][name] == hashlib.sha256(source_bytes).hexdigest()
    before_manifest = (root / 'manifest.json').read_bytes()
    before_pointer = (root / 'checkpoints/latest.json').read_bytes()
    before_attempts = sorted(path.name for path in (root / 'attempts').iterdir())
    changed = tmp_path / (name.rsplit('.', 1)[1] + '.py')
    changed.write_bytes(source_bytes + b'\n# simulated installed source change\n')
    monkeypatch.setattr(module, '__file__', str(changed))
    with pytest.raises(ValueError, match='implementation differs'):
        resume(root)
    assert (root / 'manifest.json').read_bytes() == before_manifest
    assert (root / 'checkpoints/latest.json').read_bytes() == before_pointer
    assert sorted(path.name for path in (root / 'attempts').iterdir()) == before_attempts


@pytest.mark.parametrize('cleanup_failure', [False, True])
def test_failed_native_restore_cleans_up_before_releasing_lock_without_new_attempt(tmp_path, cleanup_failure):
    import json
    import torch
    from hypergan.run_controller import run_resume
    from hypergan.run_state import run_lock
    from hypergan.single_execution import SingleProcessExecution
    from hypergan.training import resume, train

    config = write_default(tmp_path / 'config')
    root = tmp_path / 'run'
    initial_threads = torch.get_num_threads()
    cleanups = []
    try:
        torch.set_num_threads(2)
        train(config, root, stop_after_steps=1)
        before_manifest = (root / 'manifest.json').read_bytes()
        before_pointer = (root / 'checkpoints/latest.json').read_bytes()
        before_attempts = sorted(path.name for path in (root / 'attempts').iterdir())

        class RestoreFailure(SingleProcessExecution):
            def restore(self, *args, **kwargs):
                restored = super().restore(*args, **kwargs)
                assert restored.step == 1
                assert torch.get_num_threads() == 1
                raise RuntimeError('primary restore failure after loading native state')

            def shutdown(self):
                with pytest.raises(RuntimeError):
                    with run_lock(root):
                        pytest.fail('restore released ownership before adapter cleanup')
                assert (root / 'manifest.json').read_bytes() == before_manifest
                super().shutdown()
                cleanups.append(torch.get_num_threads())
                if cleanup_failure:
                    raise RuntimeError('secondary restore cleanup failure')

        with pytest.raises(RuntimeError, match='primary restore failure'):
            run_resume(root, execution_factory=RestoreFailure)
        assert cleanups == [2] and torch.get_num_threads() == 2
        assert (root / 'manifest.json').read_bytes() == before_manifest
        assert (root / 'checkpoints/latest.json').read_bytes() == before_pointer
        assert sorted(path.name for path in (root / 'attempts').iterdir()) == before_attempts
        assert json.loads(before_manifest)['status'] == 'stopped'
        result = resume(root)
        assert result['status'] == 'complete' and result['attempt_index'] == 2
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(initial_threads)


def test_failing_observer_warning_handler_cannot_change_numerics_or_threads(tmp_path, monkeypatch):
    import random
    import warnings
    import numpy as np
    import torch
    from hypergan.checkpoints import read_checkpoint
    from hypergan.training import train

    def equal(left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert type(left) is type(right) and len(left) == len(right)
            for a, b in zip(left, right):
                equal(a, b)
        else:
            assert left == right

    config = write_default(tmp_path / 'config')
    initial_threads = torch.get_num_threads()
    delivered = []

    def warning_sink(message, *args, **kwargs):
        delivered.append(str(message))
        random.random()
        np.random.random(7)
        torch.rand(9)
        torch.set_num_threads(3)

    def unavailable_observer(event):
        raise RuntimeError('observer disconnected')

    try:
        torch.set_num_threads(2)
        train(config, tmp_path / 'baseline')
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            monkeypatch.setattr(warnings, 'showwarning', warning_sink)
            train(config, tmp_path / 'observed', on_event=unavailable_observer)
        assert delivered and all('observer disconnected' in message for message in delivered)
        assert torch.get_num_threads() == 2
        equal(read_checkpoint(tmp_path / 'baseline')[2], read_checkpoint(tmp_path / 'observed')[2])
    finally:
        torch.set_num_threads(initial_threads)
