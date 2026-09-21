"""Explicit local CUDA acceptance. Run directly; no CUDA availability skips."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from hypergan.checkpoints import capture_rng, restore_rng
from hypergan.config import DEFAULT, resolve_config, write_default
from hypergan.training import ReferenceTrainer


@pytest.fixture(autouse=True)
def require_cuda():
    assert torch.cuda.is_available(), 'CUDA acceptance requires a CUDA-enabled runtime and local GPU'


@pytest.mark.parametrize('kind', ['particles', 'mog', 'gaussian'])
def test_cuda_updates_move_models_data_and_random_streams(kind):
    raw = copy.deepcopy(DEFAULT)
    raw['training'].update(device='cuda:0', steps=3)
    raw['prior'] = {'kind': kind, 'args': {'z_dim': 4, **({} if kind == 'gaussian' else {'num_particles': 32})}}
    if kind == 'gaussian':
        raw['prior_regularizer']['weight'] = 0.0
    trainer = ReferenceTrainer(resolve_config(raw))
    initial = [value.detach().clone() for value in trainer.graph.parameters()]
    for _ in range(3):
        metrics, batch = trainer.update()
        assert batch['real'].device == torch.device('cuda:0')
        assert all(torch.isfinite(torch.tensor(value)) for key, value in metrics.items() if isinstance(value, float))
    assert trainer.streams['data'].device.type == 'cpu'
    assert trainer.streams['prior'].device.type == trainer.streams['penalty'].device.type == 'cuda'
    assert all(parameter.device.type == 'cuda' for parameter in trainer.graph.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(initial, trainer.graph.parameters()))
    assert trainer.opt_g.state and trainer.opt_d.state and trainer.step == 3


def test_global_cuda_rng_roundtrip_is_exact():
    torch.rand(3, device='cuda:0')
    saved = capture_rng()
    expected = torch.rand(32, device='cuda:0')
    restore_rng(saved)
    assert torch.equal(torch.rand(32, device='cuda:0'), expected)


FIXTURE = '''
import random
import numpy as np
import torch
from hypergan.recipes import MLP

class StochasticGenerator(MLP):
    def forward(self,x):
        result=super().forward(x)
        return result + torch.rand_like(result)*.01 + (random.random()+float(np.random.random()))*.001

class Data:
    def __init__(self):
        self.cursor=0
        self.order=None
    def __call__(self,batch_size,*,generator):
        rows=[]
        while len(rows)<batch_size:
            if self.order is None or self.cursor == len(self.order):
                self.order=torch.randperm(17,generator=generator)
                self.cursor=0
            rows.append(int(self.order[self.cursor]))
            self.cursor+=1
        rows=torch.tensor(rows,dtype=torch.float32)
        return {'real':torch.stack([rows/17,-rows/17],dim=1)}
    def resume_identity(self):
        # Identity observation must not consume training CUDA randomness.
        torch.rand((),device='cuda:0')
        return {'fixture':'seventeen-items'}
    def state_dict(self):
        return {'cursor':self.cursor,'order':self.order}
    def load_state_dict(self,state):
        self.cursor,self.order=state['cursor'],state['order']
'''

DRIVER = '''
import json
from pathlib import Path
import sys
import torch
from hypergan.training import train,resume
from hypergan.checkpoints import read_checkpoint

if __name__=='__main__':
    config,run,mode=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    def observer(row):
        torch.rand(9,device='cuda:0')
    if mode=='full':
        result=train(config,run)
    elif mode=='split':
        result=train(config,run,stop_after_steps=2,preview_every=1,on_event=observer)
    else:
        result=resume(run,preview_every=1,on_event=observer)
    _,metadata,state=read_checkpoint(run)
    assert metadata['runtime']['device']=='cuda:0'
    assert 'cuda' in state['rng']
    assert state['step']==result['steps']
    torch.save(state,run/(mode+'-state.pt'))
    sample=json.loads(Path(result['sample_path']).read_text())
    assert sample['count']==8 and sample['shape']==[8,2]
'''


def _same(left,right):
    if isinstance(left,torch.Tensor):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:
            _same(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right):
            _same(a,b)
    else:
        assert left==right


def test_fresh_process_cuda_recovery_and_observation_preserve_complete_state(tmp_path):
    (tmp_path/'cuda_fixture.py').write_text(FIXTURE)
    driver=tmp_path/'driver.py'
    driver.write_text(DRIVER)
    config=write_default(tmp_path/'project',device='cuda:0')
    config.write_text(config.read_text().replace('steps = 5','steps = 4').replace('num_particles = 20000','num_particles = 32')
        .replace('count = 256','count = 8').replace('factory = "hndl"','factory = "cuda_fixture:StochasticGenerator"',1)
        .replace('factory = "gaussian_grid"','factory = "cuda_fixture:Data"').replace('side = 10\nnoise = 0.015\n',''))
    for name,mode in [('full','full'),('resumed','split'),('resumed','resume')]:
        bootstrap='import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name="__main__")'
        result=subprocess.run([sys.executable,*(['-I'] if sys.flags.isolated else []),'-c',bootstrap,str(tmp_path),str(driver),str(config),str(tmp_path/name),mode],
                              capture_output=True,text=True,timeout=60,env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8'})
        assert result.returncode==0,result.stdout+result.stderr
    _same(torch.load(tmp_path/'full/full-state.pt',weights_only=True),torch.load(tmp_path/'resumed/resume-state.pt',weights_only=True))
    reload_code='import sys;sys.path.insert(0,sys.argv[1]);from hypergan.artifacts import sample;import torch;sample(sys.argv[2],count=4);assert not torch.cuda.is_initialized()'
    result=subprocess.run([sys.executable,*(['-I'] if sys.flags.isolated else []),'-c',reload_code,str(tmp_path),str(tmp_path/'full')],
                          capture_output=True,text=True,timeout=30,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    assert result.returncode==0,result.stdout+result.stderr


def test_controlled_cpu_cuda_complete_update_parity():
    raw=copy.deepcopy(DEFAULT)
    raw['prior']['args']={'num_particles':32,'z_dim':4}
    raw['components']={
        'generator':{'factory':'linear','args':{'in_features':4,'out_features':2,'bias':False},'inputs':{'input':'latent'}},
        'discriminator':{'factory':'linear','args':{'in_features':2,'out_features':1,'bias':False},'inputs':{'input':'candidate'}}}
    cpu=ReferenceTrainer(resolve_config(raw))
    raw['training']['device']='cuda:0'
    gpu=ReferenceTrainer(resolve_config(raw))
    for name in ('graph','prior','ema_graph','ema_prior'):
        getattr(gpu,name).load_state_dict(getattr(cpu,name).state_dict())
    batch={'real':torch.randn(16,2,generator=torch.Generator().manual_seed(13))}
    ids=torch.arange(16)%7
    for _ in range(3):
        cpu.update(batch,(cpu.prior(ids),ids))
        gpu.update(batch,(gpu.prior(ids.to('cuda:0')),ids.to('cuda:0')))
    for name in ('graph','prior','ema_graph','ema_prior'):
        for left,right in zip(getattr(cpu,name).parameters(),getattr(gpu,name).parameters()):
            torch.testing.assert_close(left,right.cpu(),rtol=3e-5,atol=3e-6)
    for left,right in zip((cpu.opt_g,cpu.opt_d),(gpu.opt_g,gpu.opt_d)):
        assert len(left.state)==len(right.state)
        for a,b in zip(left.state.values(),right.state.values()):
            for key in a:
                torch.testing.assert_close(a[key],b[key].cpu(),rtol=3e-5,atol=3e-6)


def test_cuda_restore_requires_complete_cuda_rng_before_model_load():
    from hypergan.checkpoints import trainer_state, restore_trainer
    raw=copy.deepcopy(DEFAULT)
    raw['training']['device']='cuda:0'
    raw['prior']['args']['num_particles']=32
    trainer=ReferenceTrainer(resolve_config(raw))
    _,batch=trainer.update()
    state=copy.deepcopy(trainer_state(trainer,batch))
    state['rng'].pop('cuda')
    with pytest.raises(ValueError,match='missing complete CUDA RNG'):
        restore_trainer(trainer,state)


def test_explicit_cpu_stays_cold_with_cuda_available(tmp_path):
    code='''
from pathlib import Path
import sys
import torch
from hypergan.config import write_default
from hypergan.training import train
from hypergan.artifacts import sample
assert not torch.cuda.is_initialized()
root=Path(sys.argv[1])
config=write_default(root/'project',device='cpu')
result=train(config,root/'run',steps=1,preview_every=1)
sample(root/'run',count=3)
assert not torch.cuda.is_initialized(), 'Explicit CPU run initialized CUDA'
'''
    result=subprocess.run([sys.executable,*(['-I'] if sys.flags.isolated else []),'-c',code,str(tmp_path)],
                          capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
