"""Real spawned lifecycle, isolated producer, and preflight behavior."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest

from hypergan.event_views import MapSpec, read_projection_page
from hypergan.metrics import digest
from hypergan.run_state import atomic_json
from hypergan.web_autostart import Viewer, training_viewer
from hypergan.web_launch import bind_loopback


def until(predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.05)
    raise AssertionError('condition timed out')


def make_run(root):
    root.mkdir()
    definition = dict(kind='scalar', source='g_loss', label='Generator')
    definition['definition_hash'] = digest(definition)
    catalog = dict(schema_version=1, metrics={'loss/g_total': definition})
    revision = digest(catalog)
    atomic_json(root / 'metrics' / f'catalog-{revision}.json', catalog)
    rows = [dict(schema_version=2, run_id='r', attempt_id='a', stream_id='training',
                 stream_generation='r', sequence=i+1, step=i, catalog=revision,
                 event='start' if i == 0 else 'train',
                 **({'parent_attempt_id': None, 'restored_step': 0} if i == 0 else {'metrics': {'loss/g_total': float(i)}}))
            for i in range(251)]
    (root / 'events.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    atomic_json(root / 'manifest.json', dict(schema_version=1, run_id='r', attempt_id='a',
                                          steps=250, status='running', metrics_catalog=revision))


def test_pending_start_does_not_create_run_then_separate_producer_catches_up(tmp_path):
    root = tmp_path / 'run'
    with training_viewer(root, required=True) as viewer:
        assert not root.exists()
        assert viewer.info['processes']['server'] != viewer.info['processes']['projector']
        assert len({os.getpid(), viewer.process.pid, *viewer.info['processes'].values()}) == 4
        credentials = json.loads(viewer.credential_path.read_text())
        assert credentials['origin'] == viewer.session.origin
        make_run(root)
        def projected():
            try:
                page = read_projection_page(root, MapSpec().revision, limit=1000)
                return page if len(page['frames']) == 251 else None
            except (OSError, ValueError):
                return None
        page = until(projected)
        assert page['frames'][-1]['projection_sequence'] == 251
        until(lambda: list((root / 'observations').glob('viewer-*.json')))
    assert not viewer.process.is_alive()
    assert not viewer.credential_path.exists()
    receipt = json.loads(next((root / 'observations').glob('viewer-*.json')).read_text())
    assert receipt['status'] == 'stopped'
    assert credentials['token'] not in json.dumps(receipt)
    with socket.socket() as client:
        assert client.connect_ex(('127.0.0.1', int(viewer.session.host.rsplit(':', 1)[1]))) != 0


def test_default_does_not_wait_and_optional_bind_failure_is_independent(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Viewer, 'wait_ready', lambda *args: pytest.fail('automatic startup waited'))
    with training_viewer(tmp_path / 'pending'):
        pass
    with bind_loopback() as occupied:
        port = occupied.getsockname()[1]
        with training_viewer(tmp_path / 'pending', port=port) as viewer:
            assert viewer is None
        with pytest.raises(OSError):
            with training_viewer(tmp_path / 'pending', required=True, port=port):
                pytest.fail('numerical work started after failed preflight')
    assert 'training continues headless' in capsys.readouterr().err


def test_viewer_crash_does_not_fail_training_and_exception_cleans_up(tmp_path):
    with pytest.raises(ValueError, match='numerical failure'):
        with training_viewer(tmp_path / 'pending', required=True) as viewer:
            os.kill(viewer.info['processes']['server'], signal.SIGTERM)
            until(lambda: not __import__('hypergan.web_autostart', fromlist=['_probe'])._probe(viewer.session))
            assert viewer.process.is_alive()
            raise ValueError('numerical failure')
    assert not viewer.process.is_alive()
    assert not viewer.credential_path.exists()


def test_killed_parent_cleans_up_server_and_credentials(tmp_path):
    script = tmp_path / 'parent.py'
    info = tmp_path / 'info.json'
    script.write_text('''from hypergan.web_autostart import training_viewer
import json, time
from pathlib import Path
if __name__ == '__main__':
    with training_viewer(Path(__file__).parent / 'pending', required=True) as viewer:
        Path(__file__).with_name('info.json').write_text(json.dumps(viewer.info))
        time.sleep(60)
''')
    process = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        until(info.exists)
        record = json.loads(info.read_text())
        process.kill()
        process.wait(5)
        until(lambda: not Path(record['session_file']).exists())
        port = int(record['origin'].rsplit(':', 1)[1])
        def disconnected():
            with socket.socket() as client:
                return client.connect_ex(('127.0.0.1', port)) != 0
        until(disconnected)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(5)
        process.stderr.close()


def test_cli_explicit_bind_preflight_before_numerical_import_or_run_creation(tmp_path, monkeypatch, capsys):
    import builtins
    from hypergan.cli import main
    from hypergan.config import write_default
    config = write_default(tmp_path / 'project', device='cpu')
    root = tmp_path / 'run'
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name not in ('training', 'hypergan.training', 'torch'), 'numerical import before preflight'
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    with bind_loopback() as occupied:
        assert main(['train', str(config), '--run-dir', str(root), '--server-port',
                     str(occupied.getsockname()[1])]) == 1
    assert not root.exists()
    assert capsys.readouterr().out == ''
