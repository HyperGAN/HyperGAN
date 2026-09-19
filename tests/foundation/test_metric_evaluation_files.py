"""File publication recovery and bounds need no numerical dependencies."""
import hashlib
import io
import json

import pytest

from hypergan.metric_evaluation import _copy_snapshot, _recover_abandoned
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


def test_growing_snapshot_copy_never_exceeds_byte_budget(monkeypatch):
    import hypergan.metric_evaluation as module
    monkeypatch.setattr(module,'MAX_SNAPSHOT_BYTES',1024)
    output=io.BytesIO()
    with pytest.raises(ValueError,match='copy budget'):
        _copy_snapshot(io.BytesIO(b'x'*1025),output)
    assert len(output.getvalue())<=1024
    output=io.BytesIO()
    assert _copy_snapshot(io.BytesIO(b'x'*1024),output)==1024


def test_unregistered_completed_receipt_corruption_is_not_silently_repaired(tmp_path):
    directory,receipt=pending(tmp_path)
    receipt.update(status='complete',event_sha256='d'*64)
    atomic_json(directory/'receipt.json',receipt)
    (directory/'events.jsonl').write_text('{"run_id":"run"}\n')
    with pytest.raises(ValueError,match='integrity'):
        _recover_abandoned(tmp_path,'run')
    assert not (directory/'stream.json').exists()
