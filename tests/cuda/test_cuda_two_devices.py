"""Explicit cuda:1 and ambient device isolation on the authorized two-GPU host."""
import copy
import torch
from hypergan.artifacts import sample
from hypergan.checkpoints import capture_rng
from hypergan.config import DEFAULT,resolve_config
from hypergan.single_execution import SingleProcessExecution
from hypergan.previews import render_preview


def _switch_device(module,args):
    torch.cuda.set_device(0)
    torch.rand(3,device='cuda:0')
    torch.rand(5,device='cuda:1')


def test_second_device_training_and_observation_restore_ambient_device(tmp_path, monkeypatch):
    assert torch.cuda.device_count()>=2, 'This acceptance case requires the authorized two-GPU host'
    previous=torch.cuda.current_device()
    torch.cuda.set_device(0)
    raw=copy.deepcopy(DEFAULT)
    raw['training'].update(device='cuda:1',steps=2)
    raw['prior']['args']['num_particles']=32
    raw['sampling']['count']=4
    execution=SingleProcessExecution(resolve_config(raw))
    try:
        info=execution.start()
        assert info.environment['runtime']['device']=='cuda:1'
        assert torch.cuda.current_device()==1
        execution.update()
        trainer=execution._trainer
        assert next(trainer.graph.parameters()).device==torch.device('cuda:1')
        assert trainer.streams['prior'].device==trainer.streams['penalty'].device==torch.device('cuda:1')
        handle=trainer.ema_graph.models['generator'].register_forward_pre_hook(_switch_device)
        before=capture_rng()
        render_preview(trainer,execution._last_batch,{'sample_sequence':1})
        assert torch.cuda.current_device()==1
        assert all(torch.equal(a,b) for a,b in zip(before['cuda'],capture_rng()['cuda']))
        handle.remove()
        directory=tmp_path/'inference'
        directory.mkdir()
        execution.inference(directory,{'sample_sequence':2})
        import hypergan.artifacts as artifacts
        constructor=artifacts.ComponentGraph
        def switching_constructor(*args,**kwargs):
            _switch_device(None,None)
            return constructor(*args,**kwargs)
        monkeypatch.setattr(artifacts,'ComponentGraph',switching_constructor)
        before=capture_rng()
        sample(directory,count=4)
        assert torch.cuda.current_device()==1
        assert all(torch.equal(a,b) for a,b in zip(before['cuda'],capture_rng()['cuda']))
        execution.observe(lambda event:_switch_device(None,None),{})
        assert torch.cuda.current_device()==1
    finally:
        execution.shutdown()
    assert torch.cuda.current_device()==0
    torch.cuda.set_device(previous)
