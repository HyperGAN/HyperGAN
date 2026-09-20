"""Custom scalar isolation, source changes, errors and fresh worker cleanup."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from hypergan.config import write_default

PLUGIN = '''
import ctypes
import os
from pathlib import Path
import random

class Probe:
    def __init__(self, marker=None, fail=False, slow=False):
        self.marker, self.fail, self.slow = marker, fail, slow
    def describe(self):
        return {'kind':'scalar','label':'Probe','unit':'ratio','direction':'none'}
    def evaluate(self, *, numerator, denominator, context):
        if self.marker:
            with Path(self.marker).open('a') as stream: stream.write(str(os.getpid())+'\\n')
        random.random()
        import numpy as np
        import torch
        assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
        assert torch.get_num_threads() == 1
        np.random.rand()
        torch.rand(9)
        if self.slow: ctypes.PyDLL(None).sleep(30)
        if self.fail: raise ValueError('explicit probe failure')
        context.clear()
        return numerator / denominator
'''
DRIVER = '''
import json
from pathlib import Path
import sys
from hypergan.training import train, resume
from hypergan.checkpoints import read_checkpoint
import torch
if __name__ == '__main__':
    root, mode = Path(sys.argv[1]), sys.argv[2]
    if mode == 'normal':
        train(root/'base.toml', root/'base')
        train(root/'custom.toml', root/'custom')
        for name in ('base','custom'):
            torch.save(read_checkpoint(root/name)[2],root/(name+'.pt'))
    elif mode == 'recover':
        train(root/'base.toml',root/'base')
        resume(root/'custom',config_path=root/'base.toml')
        for name in ('base','custom'):
            torch.save(read_checkpoint(root/name)[2],root/(name+'.pt'))
    elif mode == 'split':
        train(root/'custom.toml',root/'custom',stop_after_steps=1)
    elif mode == 'resume':
        resume(root/'custom')
    else:
        train(root/'custom.toml',root/'custom')
'''


def setup(tmp_path, *, args='', timeout=5, cadence=2, on_error='disable'):
    (tmp_path/'metric_probe.py').write_text(PLUGIN)
    driver = tmp_path/'driver.py'
    driver.write_text(DRIVER)
    config = write_default(tmp_path/'base.toml',device='cpu')
    base = config.read_text().replace('steps = 5','steps = 3').replace('num_particles = 20000','num_particles = 32')
    config.write_text(base)
    (tmp_path/'custom.toml').write_text(base+f'''
[metrics.custom.ratio]
factory = "metric_probe:Probe"
inputs = {{ numerator = "update.g_loss", denominator = "update.d_loss" }}
every_steps = {cadence}
timeout = {timeout}
on_error = "{on_error}"
[metrics.custom.ratio.args]
{args}
''')
    return driver


def run(driver, mode):
    bootstrap = 'import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name="__main__")'
    return subprocess.run([sys.executable,'-c',bootstrap,str(driver.parent),str(driver),str(driver.parent),mode],capture_output=True,text=True,timeout=60)


def rows(path):
    return [json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]


def same(left,right):
    if isinstance(left,torch.Tensor): torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left: same(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right): same(a,b)
    else: assert left==right


def test_supervised_scalar_cadence_and_rng_do_not_change_complete_state(tmp_path):
    marker=tmp_path/'pids'
    driver=setup(tmp_path,args='marker = '+json.dumps(str(marker)))
    result=run(driver,'normal')
    assert result.returncode==0,result.stdout+result.stderr
    same(torch.load(tmp_path/'base.pt',weights_only=True),torch.load(tmp_path/'custom.pt',weights_only=True))
    events=rows(tmp_path/'custom')
    training=[row for row in events if row['event']=='train']
    measured=[row for row in events if 'ratio' in row.get('metrics',{})]
    assert [row['step'] for row in measured]==[2]
    assert measured[0]['event']=='metric'
    assert measured[0]['metrics']['ratio']==training[1]['metrics']['loss/g_total']/training[1]['metrics']['loss/d_total']
    for pid in map(int,marker.read_text().splitlines()):
        with pytest.raises(ProcessLookupError): os.kill(pid,0)


@pytest.mark.parametrize('slow',[False,True])
def test_optional_failure_and_timeout_disable_metric_and_reap_worker(tmp_path,slow):
    marker=tmp_path/'pids'
    driver=setup(tmp_path,args='marker = '+json.dumps(str(marker))+('\nslow = true' if slow else '\nfail = true'),timeout=5,cadence=1)
    result=run(driver,'error')
    assert result.returncode==0,result.stdout+result.stderr
    training=[row for row in rows(tmp_path/'custom') if row['event']=='train']
    assert len(training)==3
    assert training[0]['measurement_status']['ratio']['status']=='queued'
    assert all(row['measurement_status']['ratio']['status'] in ('dropped','disabled') for row in training[1:])
    assert any(row.get('measurement_status',{}).get('ratio',{}).get('status')=='disabled'
               for row in rows(tmp_path/'custom'))
    assert len(marker.read_text().splitlines())==1
    for pid in map(int,marker.read_text().splitlines()):
        with pytest.raises(ProcessLookupError): os.kill(pid,0)


def test_required_factory_failure_is_not_successful_training(tmp_path):
    driver=setup(tmp_path,args='fail = true',cadence=1,on_error='fail')
    result=run(driver,'error')
    assert result.returncode!=0 and 'Required metric ratio failed' in result.stderr
    manifest=json.loads((tmp_path/'custom/manifest.json').read_text())
    assert manifest['status']=='failed' and 1 <= manifest['steps'] <= 3
    assert manifest['last_durable_step'] < manifest['steps']
    assert [row for row in rows(tmp_path/'custom') if row['event']=='train']
    result=run(driver,'recover')
    assert result.returncode==0,result.stdout+result.stderr
    same(torch.load(tmp_path/'base.pt',weights_only=True),torch.load(tmp_path/'custom.pt',weights_only=True))


def test_factory_source_revision_changes_catalog_but_allows_numerical_resume(tmp_path):
    driver=setup(tmp_path,cadence=1)
    result=run(driver,'split')
    assert result.returncode==0,result.stderr
    old=json.loads((tmp_path/'custom/manifest.json').read_text())['metrics_catalog']
    plugin=tmp_path/'metric_probe.py'
    plugin.write_text(plugin.read_text().replace('return numerator / denominator','return 2 * numerator / denominator'))
    result=run(driver,'resume')
    assert result.returncode==0,result.stderr
    new=json.loads((tmp_path/'custom/manifest.json').read_text())['metrics_catalog']
    assert old!=new
    a=json.loads((tmp_path/f'custom/metrics/catalog-{old}.json').read_text())
    b=json.loads((tmp_path/f'custom/metrics/catalog-{new}.json').read_text())
    assert a['metrics']['ratio']['definition_hash']!=b['metrics']['ratio']['definition_hash']


def test_replicated_custom_scalar_keeps_parent_torch_free_and_state_exact(tmp_path):
    driver=setup(tmp_path,cadence=2,timeout=10)
    driver.write_text('''
import json
from pathlib import Path
import sys
from hypergan.replicated_execution import run_train
if __name__=='__main__':
    root=Path(sys.argv[1])
    profile={'schema_version':1,'execution':{'name':'cpu-replicated-gloo','world_size':2,'accumulation_steps':2}}
    for name,config in [('base','base.toml'),('custom','custom.toml')]:
        run_train(root/config,root/name,profile=profile)
        assert 'torch' not in sys.modules,'Custom metric imported runtime in distributed coordinator'
''')
    result=run(driver,'normal')
    assert result.returncode==0,result.stdout+result.stderr
    def states(name):
        root=tmp_path/name/'distributed-checkpoints'
        checkpoint=json.loads((root/'latest.json').read_text())['checkpoint']
        return [torch.load(root/checkpoint/f'rank-{rank:05d}.pt',weights_only=True) for rank in range(2)]
    same(states('base'),states('custom'))
    assert [row['step'] for row in rows(tmp_path/'custom') if 'ratio' in row.get('metrics',{})]==[2]


@pytest.mark.parametrize('factory', ['missing_metric_module:Probe','hypergan.metric_examples:ColorHistogramDifference'])
def test_missing_factory_or_incompatible_descriptor_fails_before_run_mutation(tmp_path,factory):
    driver=setup(tmp_path)
    config=tmp_path/'custom.toml'
    config.write_text(config.read_text().replace('metric_probe:Probe',factory))
    result=run(driver,'error')
    assert result.returncode!=0
    assert not (tmp_path/'custom').exists()


def test_async_cancel_reaps_native_blocked_scalar_without_waiting_for_timeout(tmp_path):
    driver=setup(tmp_path, timeout=30)
    driver.write_text('''
from pathlib import Path
import os
import time
import sys
from hypergan.config import load_config
from hypergan.metric_plugins import ScalarMetrics, prepare_custom

if __name__ == '__main__':
    root=Path(sys.argv[1])
    config=load_config(root/'custom.toml')
    spec=config['metrics']['custom']['ratio']
    spec['args']={'slow':True,'marker':str(root/'worker-pid')}
    spec['every_steps']=1
    prepare_custom(config)
    metrics=ScalarMetrics(config)
    metrics.start()
    started=time.monotonic()
    metrics.evaluate({'g_loss':2.,'d_loss':1.},{'step':1})
    for step in range(2,1002):
        assert metrics.evaluate({'g_loss':2.,'d_loss':1.},{'step':step})[1]['ratio']['status']=='dropped'
        assert metrics.poll()==[]
    elapsed=time.monotonic()-started
    assert elapsed<1, elapsed
    deadline=time.monotonic()+15
    while not (root/'worker-pid').exists() and time.monotonic()<deadline:
        time.sleep(.01)
    pid=int((root/'worker-pid').read_text())
    started=time.monotonic()
    metrics.close(drain=False)
    elapsed=time.monotonic()-started
    assert elapsed<8, elapsed
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError('Metric worker survived cancellation')
''')
    result=run(driver,'cancel')
    assert result.returncode==0,result.stdout+result.stderr
