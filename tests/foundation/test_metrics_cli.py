import json
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
    assert 'torch' not in sys.modules


def test_missing_projection_is_actionable(tmp_path, capsys):
    assert main(['contributions', str(tmp_path)]) == 1
    assert 'error:' in capsys.readouterr().err
    assert not (tmp_path / 'views' / MapSpec().revision).exists()
