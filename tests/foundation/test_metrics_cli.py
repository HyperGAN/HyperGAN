import json
import subprocess
import sys

from hypergan.cli import main
from hypergan.config import resolve_config
from hypergan.event_views import MapSpec
from hypergan.metrics import publish_catalog


def test_catalog_project_and_exact_pages_without_training(tmp_path, capsys):
    revision = publish_catalog(tmp_path, resolve_config({}))
    (tmp_path / 'manifest.json').write_text(json.dumps({'metrics_catalog': revision}))
    events = [dict(schema_version=2, event='train', run_id='r', stream_id='training',
                   stream_generation='r', attempt_id='a', sequence=i, step=i,
                   catalog=revision, metrics={'loss/d_total': float(i)}) for i in range(1, 5)]
    (tmp_path / 'events.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in events))
    assert main(['metrics', str(tmp_path)]) == 0
    assert 'loss/d_total' in json.loads(capsys.readouterr().out)['metrics']
    assert main(['project', str(tmp_path), '--limit', '2']) == 0
    assert json.loads(capsys.readouterr().out)['projection_sequence'] == 4
    assert main(['contributions', str(tmp_path), '--limit', '2']) == 0
    first = json.loads(capsys.readouterr().out)
    assert len(first['frames']) == 2
    assert main(['contributions', str(tmp_path), '--cursor', first['cursor']]) == 0
    second = json.loads(capsys.readouterr().out)
    assert [f['projection_sequence'] for f in second['frames']] == [3, 4]
    assert main(['project', str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)['documents'] == 0
    # Reference test collection may already import Torch in this pytest
    # process. Prove the CLI import boundary in a fresh interpreter, including
    # real projection work rather than only reopening the existing projection.
    code = """
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import sys
assert 'torch' not in sys.modules
from hypergan.cli import main
source = Path(sys.argv[1])
root = source / 'isolated-cli'
root.mkdir()
shutil.copytree(source / 'metrics', root / 'metrics')
for name in ('manifest.json', 'events.jsonl'):
    shutil.copyfile(source / name, root / name)
outputs = []
for command in ('metrics', 'project', 'contributions'):
    output = StringIO()
    with redirect_stdout(output):
        assert main([command, str(root)]) == 0
    outputs.append(json.loads(output.getvalue()))
    assert 'torch' not in sys.modules, command
assert 'loss/d_total' in outputs[0]['metrics']
assert outputs[1]['documents'] == 4
assert [frame['projection_sequence'] for frame in outputs[2]['frames']] == [1, 2, 3, 4]
"""
    result = subprocess.run([sys.executable, '-I', '-c', code, str(tmp_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_projection_is_actionable(tmp_path, capsys):
    assert main(['contributions', str(tmp_path)]) == 1
    assert 'error:' in capsys.readouterr().err
    assert not (tmp_path / 'views' / MapSpec().revision).exists()
