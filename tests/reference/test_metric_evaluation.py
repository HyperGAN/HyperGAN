"""Manual immutable snapshot evaluation with independent data/RNG and streams."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from hypergan.config import write_default
from hypergan.metric_examples import ColorMomentDistance, ColorHistogramDifference

FIXTURE = '''
import torch
from hypergan.recipes import MLP
class RGBGenerator(MLP):
    def forward(self,x): return super().forward(x).tanh().reshape(-1,3,2,2)
class RGBDiscriminator(MLP):
    def forward(self,x): return super().forward(x.flatten(1))
class RGBData:
    def __call__(self,batch_size,*,generator):
        return {'real': torch.rand(batch_size,3,2,2,generator=generator)*2-1}
    def resume_identity(self): return {'dataset':'rgb-fixture','version':1,'split':'evaluation'}
    def state_dict(self): return {}
    def load_state_dict(self,state): assert state=={}
class PartialMetric:
    def describe(self): return {'kind':'scalar'}
    def evaluate(self,*,batches,context): next(iter(batches)); return 0.0
'''
DRIVER = '''
import hashlib
import json
from pathlib import Path
import sys
from hypergan.training import train,resume
from hypergan.metric_evaluation import evaluate
if __name__=='__main__':
    root, mode = Path(sys.argv[1]), sys.argv[2]
    if mode=='train':
        stopped=train(root/'config.toml',root/'run',stop_after_steps=1)
        (root/'stopped.json').write_text(json.dumps(stopped))
        resume(root/'run')
    elif mode=='evaluate':
        stopped=json.loads((root/'stopped.json').read_text())
        a=evaluate(root/'run','mean',bundle=stopped['bundle_path'])
        b=evaluate(root/'run','mean',bundle=stopped['bundle_path'])
        c=evaluate(root/'run','histogram')
        (root/'results.json').write_text(json.dumps([a,b,c]))
    else:
        evaluate(root/'run','partial',config_path=root/'partial.toml')
'''


def setup(tmp_path):
    (tmp_path/'eval_fixture.py').write_text(FIXTURE)
    driver=tmp_path/'evaluate_driver.py'
    driver.write_text(DRIVER)
    path=write_default(tmp_path/'config.toml',device='cpu')
    text=path.read_text().replace('steps = 5','steps = 2').replace('num_particles = 20000','num_particles = 32').replace('count = 256','count = 4')
    text=text.replace('factory = "mlp"','factory = "eval_fixture:RGBGenerator"',1).replace('factory = "mlp"','factory = "eval_fixture:RGBDiscriminator"',1)
    text=text.replace('output_dim = 2','output_dim = 12').replace('input_dim = 2','input_dim = 12')
    text=text.replace('factory = "gaussian_grid"','factory = "eval_fixture:RGBData"').replace('side = 10\nnoise = 0.015\n','')
    text+='\n[training.backend]\ndeterministic_algorithms = true\n'
    for name,factory in [('mean','hypergan.metric_examples:ColorMomentDistance'),('histogram','hypergan.metric_examples:ColorHistogramDifference')]:
        text+=metric(name,factory)
    path.write_text(text)
    (tmp_path/'partial.toml').write_text(text+metric('partial','eval_fixture:PartialMetric'))
    return driver


def metric(name,factory):
    return f'''
[metrics.custom.{name}]
factory = "{factory}"
mode = "snapshot"
trigger = "manual"
timeout = 20
inputs = {{ generated = "evaluation.generated", reference = "evaluation.reference" }}
[metrics.custom.{name}.evaluation]
device = "cpu"
sample_count = 7
batch_size = 3
seed = 88
[metrics.custom.{name}.evaluation.data]
factory = "eval_fixture:RGBData"
args = {{}}
'''


def run(driver,mode):
    bootstrap='import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name="__main__")'
    return subprocess.run([sys.executable,'-c',bootstrap,str(driver.parent),str(driver),str(driver.parent),mode],capture_output=True,text=True,timeout=90)


def test_color_moments_histogram_reference_and_pixel_weighting():
    batches=[{'generated': torch.zeros(1,3,2,2), 'reference': torch.ones(1,3,2,2)},
             {'generated': torch.ones(2,3,2,2), 'reference': torch.ones(2,3,2,2)}]
    assert ColorMomentDistance(low=0,high=1).evaluate(batches=iter(batches),context={})==pytest.approx(1/3)
    assert ColorMomentDistance(statistic='spread',low=0,high=1).evaluate(batches=iter(batches),context={})==pytest.approx((2/9)**.5)
    hist=ColorHistogramDifference(bins=2,low=0,high=1).evaluate(batches=iter(batches),context={})
    assert hist['edges']==[0,.5,1] and hist['counts']==pytest.approx([1/3,1/3])
    with pytest.raises(ValueError,match='RGB'):
        ColorMomentDistance().evaluate(batches=iter([{'generated':torch.zeros(1,2),'reference':torch.zeros(1,2)}]),context={})


def test_histogram_detaches_and_preserves_strict_backend_policy(monkeypatch):
    original = torch.histc
    calls = []

    def detached_histogram(value, **kwargs):
        assert not value.requires_grad
        assert value.device.type == 'cpu'
        calls.append(value.numel())
        return original(value, **kwargs)

    monkeypatch.setattr(torch, 'histc', detached_histogram)
    enabled, warn_only = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        batches = [{'generated': torch.zeros(1,3,2,2,requires_grad=True),
                    'reference': torch.ones(1,3,2,2,requires_grad=True)}]
        result = ColorHistogramDifference(bins=2,low=0,high=1).evaluate(batches=batches,context={})
        assert result == {'edges':[0.0,0.5,1.0], 'counts':[1.0,1.0]}
        assert calls == [12,12]
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_snapshot_is_pinned_repeatable_and_publishes_independent_late_stream(tmp_path):
    driver=setup(tmp_path)
    result=run(driver,'train')
    assert result.returncode==0,result.stdout+result.stderr
    before=(tmp_path/'run/events.jsonl').read_bytes()
    manifest=(tmp_path/'run/manifest.json').read_bytes()
    checkpoints={path:path.read_bytes() for path in (tmp_path/'run/checkpoints').rglob('*') if path.is_file()}
    result=run(driver,'evaluate')
    assert result.returncode==0,result.stdout+result.stderr
    a,b,c=json.loads((tmp_path/'results.json').read_text())
    assert a['result']['value']==b['result']['value']
    assert a['catalog']==b['catalog'] and a['evaluation_id']!=b['evaluation_id']
    assert a['result']['step']==1 and c['result']['step']==2
    assert a['result']['protocol']['data_identity']['split']=='evaluation'
    assert a['result']['protocol']['evaluation']['sample_count']==7
    runtime = a['result']['protocol']['runtime']
    assert runtime['configured_training_backend'] == {'deterministic_algorithms': True}
    assert runtime['backend']['deterministic_algorithms'] is True
    assert runtime['backend']['deterministic_warn_only'] is False
    for receipt in (a,b,c):
        directory=tmp_path/'run/metrics/evaluations'/receipt['evaluation_id']
        assert not (directory/'snapshot.pt').exists()
        event=json.loads((directory/'events.jsonl').read_text())
        registry=json.loads((directory/'stream.json').read_text())
        assert registry['stream_id']==event['stream_id']
        assert event['snapshot_identity']['attempt_id']==receipt['result']['snapshot_identity']['attempt_id']
        assert event['step']==receipt['result']['step']
    assert len(json.loads((tmp_path/'run/metrics/evaluations'/c['evaluation_id']/'events.jsonl').read_text())['distributions']['histogram']['counts'])==32
    assert (tmp_path/'run/events.jsonl').read_bytes()==before
    assert (tmp_path/'run/manifest.json').read_bytes()==manifest
    assert all(path.read_bytes()==data for path,data in checkpoints.items())


def test_partial_sample_consumption_fails_with_visible_receipt(tmp_path):
    driver=setup(tmp_path)
    result=run(driver,'train')
    assert result.returncode==0,result.stderr
    result=run(driver,'partial')
    assert result.returncode!=0 and 'complete declared evaluation sample count' in result.stderr
    receipts=list((tmp_path/'run/metrics/evaluations').glob('*/receipt.json'))
    assert len(receipts)==1
    receipt=json.loads(receipts[0].read_text())
    assert receipt['status']=='failed'
    event=json.loads((receipts[0].parent/'events.jsonl').read_text())
    assert event['metrics']=={} and event['measurement_status']['partial']['status']=='failed'
    assert (receipts[0].parent/'stream.json').exists()
