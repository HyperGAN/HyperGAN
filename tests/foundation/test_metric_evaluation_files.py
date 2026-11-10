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
