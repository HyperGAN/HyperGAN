"""Launching prepared runs: device overrides train, and launch guards hold with torch present.

The preparation halves of these checks are lightweight tests in tests/foundation.
"""
import json
import subprocess
import sys
import textwrap

import pytest

from hypergan.config import fingerprint
from hypergan.execution import prepare_resume, prepare_train
from tests.config_fixtures import write_default


def test_device_override_is_the_configuration_that_trains_and_resumes(tmp_path, monkeypatch):
    # The recipe asks for CUDA, which this CPU-only run cannot use; the override must reach the trainer.
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    path, run = write_default(tmp_path / 'project', device='cuda', defaults='hypergan'), tmp_path / 'run'
    prepared = prepare_train(path, run, devices='cpu', stop_after_steps=1)
    assert prepared.config['training']['device'] == 'cpu'
    prepared.run()
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['config']['training']['device'] == 'cpu'
    assert manifest['config_sha256'] == fingerprint(prepared.config)
    assert (run / 'config-source').is_dir()  # provenance still snapshots the source file
    again = prepare_train(path, run, devices='cpu')
    assert again.operation == 'resume' and again.config['training']['device'] == 'cpu'
    import torch
    if hasattr(torch, "cuda"):
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    migrated = prepare_resume(run, devices='0')
    assert migrated.config['training']['device'] == 'cuda'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(ValueError, match='training.device'):
        prepare_resume(run, config_path=path)


def test_device_selection_is_refused_after_cuda_initialized(tmp_path, monkeypatch):
    import torch
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.delenv('CUDA_DEVICE_ORDER', raising=False)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    path = write_default(tmp_path / 'project', device='cuda', defaults='hypergan')
    with pytest.raises(ValueError, match='CUDA is already initialized'):
        prepare_train(path, tmp_path / 'run', devices='1')
    assert 'CUDA_DEVICE_ORDER' not in __import__('os').environ  # nothing was changed
    # Hiding every GPU (devices='cpu') stays allowed: it needs no new CUDA mask.
    assert prepare_train(path, tmp_path / 'run', devices='cpu').config['training']['device'] == 'cpu'


@pytest.mark.heavy
def test_unguarded_script_with_previews_fails_before_running_twice(tmp_path):
    marker = tmp_path / 'top-level-runs.txt'
    script = tmp_path / 'unguarded.py'
    script.write_text(textwrap.dedent(f'''
        import hypergan.api as hg
        with open({str(marker)!r}, 'a') as stream:
            stream.write('ran\\n')
        model = hg.override(hg.packaged('gaussian-grid'), {{'train.device': 'cpu'}})
        hg.train(model, {str(tmp_path / 'run')!r}, steps=2, previews=1, devices='cpu')
    '''), encoding='utf-8')
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True,
                            timeout=120)
    assert result.returncode != 0
    assert 'without a main guard' in result.stderr
    assert marker.read_text() == 'ran\n'
