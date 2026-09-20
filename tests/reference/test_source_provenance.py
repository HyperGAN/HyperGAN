"""Original and current release identities survive current-run recovery."""

import json

import pytest

from hypergan.config import write_default


@pytest.mark.parametrize('existing_run', [False, True])
def test_resume_preserves_initial_source_and_records_each_attempt(tmp_path, monkeypatch, existing_run):
    from hypergan import single_execution
    from hypergan.checkpoints import read_checkpoint
    from hypergan.training import train, resume

    config = write_default(tmp_path / 'config.toml', device='cpu')
    run = tmp_path / 'run'
    original = {'hypergan_commit': 'a' * 40, 'hypergan_dirty': False, 'hypergan_provenance': 'build'}
    current = {'hypergan_commit': 'b' * 40, 'hypergan_dirty': True, 'hypergan_provenance': 'git'}
    monkeypatch.setattr(single_execution, 'source_info', lambda: original)
    first = train(config, run, steps=2, stop_after_steps=1)
    first_attempt = run / 'attempts' / first['attempt_id'] / 'manifest.json'
    first_bytes = first_attempt.read_bytes()
    _, checkpoint, _ = read_checkpoint(run)
    assert checkpoint['source'] == checkpoint['initial_source'] == original
    if existing_run:
        manifest = json.loads((run / 'manifest.json').read_text())
        del manifest['initial_source']
        (run / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(single_execution, 'source_info', lambda: current)
    second = resume(run)
    assert second['status'] == 'complete'
    assert second['source'] == current
    assert second['initial_source'] == original
    assert first_attempt.read_bytes() == first_bytes
    second_attempt = json.loads((run / 'attempts' / second['attempt_id'] / 'manifest.json').read_text())
    assert second_attempt['source'] == current
    assert second_attempt['initial_source'] == original
    _, checkpoint, _ = read_checkpoint(run)
    assert checkpoint['source'] == current
    assert checkpoint['initial_source'] == original


def test_missing_initial_source_uses_first_attempt_not_previous_release(tmp_path, monkeypatch):
    from hypergan import single_execution
    from hypergan.training import train, resume

    config = write_default(tmp_path / 'config.toml', device='cpu')
    run = tmp_path / 'run'
    source = {'hypergan_commit': 'a' * 40}
    monkeypatch.setattr(single_execution, 'source_info', lambda: dict(source))
    first = train(config, run, steps=3, stop_after_steps=1)
    source['hypergan_commit'] = 'b' * 40
    second = resume(run, stop_after_steps=1)
    assert second['source']['hypergan_commit'] == 'b' * 40
    manifest = json.loads((run / 'manifest.json').read_text())
    del manifest['initial_source']
    del manifest['initial_source_origin']
    (run / 'manifest.json').write_text(json.dumps(manifest))
    source['hypergan_commit'] = 'c' * 40
    third = resume(run)
    assert third['initial_source']['hypergan_commit'] == 'a' * 40
    assert third['source']['hypergan_commit'] == 'c' * 40
    assert third['initial_source_origin'] == f"attempts/{first['attempt_id']}/manifest.json"


def test_failed_start_records_current_source_before_execution_environment(tmp_path, monkeypatch):
    from hypergan import provenance, single_execution
    from hypergan.run_controller import run_resume
    from hypergan.training import train

    config = write_default(tmp_path / 'config.toml', device='cpu')
    run = tmp_path / 'run'
    original = {'hypergan_commit': 'a' * 40}
    current = {'hypergan_commit': 'b' * 40, 'hypergan_dirty': False, 'hypergan_provenance': 'build'}
    monkeypatch.setattr(single_execution, 'source_info', lambda: original)
    train(config, run, steps=2, stop_after_steps=1)
    monkeypatch.setattr(provenance, 'hypergan_source', lambda: current)

    class FailingStart(single_execution.SingleProcessExecution):
        def start(self):
            raise RuntimeError('start failed before environment publication')

    with pytest.raises(RuntimeError, match='start failed'):
        run_resume(run, execution_factory=FailingStart)
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert manifest['source']['hypergan_commit'] == current['hypergan_commit']
    assert manifest['initial_source'] == original
    attempt = json.loads((run / 'attempts' / manifest['attempt_id'] / 'manifest.json').read_text())
    assert attempt['source'] == manifest['source']
    assert attempt['initial_source'] == original
