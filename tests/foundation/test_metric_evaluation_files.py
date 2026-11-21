"""File publication recovery and bounds need no numerical dependencies."""
import hashlib
import json

import pytest

from hypergan.metric_evaluation import _recover_abandoned
from hypergan.run_state import atomic_json


def pending(tmp_path):
    directory=tmp_path/'metrics/evaluations'/('a'*32)
    directory.mkdir(parents=True)
    receipt={'schema_version':1,'status':'running','run_id':'run','evaluation_id':directory.name,
             'catalog':'b'*64,'snapshot_sha256':'c'*64,'metric_id':'quality'}
    atomic_json(directory/'receipt.json',receipt)
    (directory/'snapshot.pt').write_bytes(b'pinned temporary bytes')
    return directory,receipt


def test_abandoned_evaluation_becomes_visible_failed_stream_and_releases_pin(tmp_path):
    directory,_=pending(tmp_path)
    _recover_abandoned(tmp_path,'run')
    receipt=json.loads((directory/'receipt.json').read_text())
    event=json.loads((directory/'events.jsonl').read_text())
    assert receipt['status']=='failed' and event['status']=='failed'
    assert event['metrics']=={} and event['source_position_known'] is False
    assert 'new evaluation ID' in event['measurement_status']['quality']['reason']
    assert (directory/'stream.json').is_file() and not (directory/'snapshot.pt').exists()
    before=(directory/'events.jsonl').read_bytes()
    _recover_abandoned(tmp_path,'run')
    assert (directory/'events.jsonl').read_bytes()==before


def test_committed_receipt_without_registry_is_rediscovered_without_changing_result(tmp_path):
    directory,receipt=pending(tmp_path)
    event={'run_id':'run','evaluation_id':directory.name,'stream_id':'evaluation:'+directory.name,
           'metrics':{'quality':.5}}
    encoded=json.dumps(event).encode()+b'\n'
    (directory/'events.jsonl').write_bytes(encoded)
    receipt.update(status='complete',event_sha256=hashlib.sha256(encoded).hexdigest())
    atomic_json(directory/'receipt.json',receipt)
    _recover_abandoned(tmp_path,'run')
    assert (directory/'stream.json').is_file()
    assert (directory/'events.jsonl').read_bytes()==encoded
    assert json.loads((directory/'receipt.json').read_text())['status']=='complete'
    assert not (directory/'snapshot.pt').exists()


def test_unregistered_completed_receipt_corruption_is_not_silently_repaired(tmp_path):
    directory,receipt=pending(tmp_path)
    receipt.update(status='complete',event_sha256='d'*64)
    atomic_json(directory/'receipt.json',receipt)
    (directory/'events.jsonl').write_text('{"run_id":"run"}\n')
    with pytest.raises(ValueError,match='integrity'):
        _recover_abandoned(tmp_path,'run')
    assert not (directory/'stream.json').exists()


def test_live_manual_evaluation_is_not_recovered_as_abandoned(tmp_path):
    # Review 2026-10-25 slice 5 M1: training start/close used to fail an
    # in-flight manual evaluation and delete its pinned snapshot.
    from foldlitedb.fileio import file_lock
    from hypergan.metric_evaluation import OWNER_LOCK
    directory,receipt=pending(tmp_path)
    with file_lock(directory,name=OWNER_LOCK):
        _recover_abandoned(tmp_path,'run')
        assert json.loads((directory/'receipt.json').read_text())==receipt
        assert (directory/'snapshot.pt').is_file()
        assert not (directory/'events.jsonl').exists() and not (directory/'stream.json').exists()
    # A dead owner releases the lock, and the directory is then recoverable.
    _recover_abandoned(tmp_path,'run')
    assert json.loads((directory/'receipt.json').read_text())['status']=='failed'


def test_import_scores_without_checkpoint_or_factory_is_atomic_and_idempotent(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import hypergan as hg
    from hypergan.summary import collect_evaluations

    atomic_json(tmp_path/'manifest.json', {'run_id': 'run', 'attempt_id': 'current', 'status': 'running'})
    outputs = {'x': {'label': 'Full values', 'unit': 'distance', 'direction': 'minimize'},
               'm': {'label': 'Masks', 'unit': 'score', 'direction': 'maximize'}}
    values = {'x': 2, 'm': 0.5}
    def publish(scores):
        return hg.import_evaluation(tmp_path, 'paired', outputs=outputs, values=scores, step=1000,
            snapshot_identity={'attempt_id': 'old', 'source_file': 'archived'}, snapshot_sha256='a'*64,
            protocol={'scorer': 'v2', 'sample_count': 2048})

    with ThreadPoolExecutor(max_workers=4) as writers:
        results = list(writers.map(publish, [values]*8))
    assert all(result == results[0] for result in results)
    receipt = results[0]
    assert receipt['attempt_id'] == 'old'
    assert receipt['source_step'] == 1000
    assert receipt['snapshot_sha256'] == 'a'*64
    assert receipt['result']['protocol']['sample_count'] == 2048
    with pytest.raises(ValueError, match='Conflicting'):
        publish(dict(values, m=3))
    for invalid in ({'x': 2}, dict(values, extra=1), dict(values, m=True), dict(values, x=float('nan'))):
        with pytest.raises(ValueError):
            publish(invalid)
    assert len(list((tmp_path/'metrics/evaluations').iterdir())) == 1
    assert [(r['metric'], r['unit'], r['direction']) for r in collect_evaluations(tmp_path)] == [
        ('paired/m', 'score', 'maximize'), ('paired/x', 'distance', 'minimize')]
    assert hg.evaluations(tmp_path)['paired/m'][0]['value'] == 0.5


def test_interrupted_named_evaluation_reports_every_output(tmp_path):
    directory, receipt = pending(tmp_path)
    atomic_json(directory/'receipt.json', dict(receipt, output_ids=['x', 'm'], source_step=12))
    _recover_abandoned(tmp_path, 'run')
    event = json.loads((directory/'events.jsonl').read_text())
    assert set(event['measurement_status']) == {'quality/x', 'quality/m'}
    assert event['step'] == 12 and event['metrics'] == {}


def test_interrupted_import_does_not_expose_partial_scores(tmp_path, monkeypatch):
    from pathlib import Path
    from hypergan import import_evaluation

    atomic_json(tmp_path/'manifest.json', {'run_id': 'run'})
    kwargs = dict(step=9, outputs={'x': {}, 'm': {}}, values={'x': 0, 'm': 1})
    def interrupt(*args):
        raise OSError('simulated write failure')
    with monkeypatch.context() as fault:
        fault.setattr(Path, 'rename', interrupt)
        with pytest.raises(OSError, match='simulated'):
            import_evaluation(tmp_path, 'paired', **kwargs)
    assert not list((tmp_path/'metrics/evaluations').iterdir())
    receipt = import_evaluation(tmp_path, 'paired', **kwargs)
    assert receipt['result']['values'] == kwargs['values']
    assert receipt['attempt_id'] == 'unknown'
    assert 'snapshot_sha256' not in receipt
