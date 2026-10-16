"""Develop recipe transfer, actual updates and exact resume through HyperGAN."""
import copy
import pytest
import torch
from hypergan.config import resolve_config
from hypergan.training import ReferenceTrainer
from hypergan.checkpoints import trainer_state, restore_trainer


def config(trainer):
    return resolve_config({'particlegan_trainer':trainer,
                           'prior':{'args':{'num_particles':16,'z_dim':4}},
                           'training':{'steps':4,'batch_size':4}})


def equal(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b


@pytest.mark.parametrize('trainer',['bcap','bispf'])
def test_selected_recipe_updates_and_replays_all_optimizer_state(trainer):
    c=config(trainer)
    a=ReferenceTrainer(c)
    assert a.recipe.critic_formulation==trainer and a.recipe.loss=='non_saturating'
    assert a.recipe.optimizer_family=='dualnorm' and a.recipe.constraint_geometry_mode=='direction_blend'
    assert a.recipe.lr==.012 and a.recipe.prior_lr_mult==2.5
    initial=a.prior.z.detach().clone()
    _,batch=a.update()
    state=copy.deepcopy(trainer_state(a,batch))
    assert not torch.equal(initial,a.prior.z)
    assert 'direction_blend' in state['optimizers'][0]
    b=ReferenceTrainer(c)
    restore_trainer(b,state)
    row_a,_=a.update(); row_b,_=b.update()
    equal(row_a,row_b)
    for name in ('graph','prior'):
        equal(getattr(a,name).state_dict(),getattr(b,name).state_dict())
    equal(a.opt_g.state_dict(),b.opt_g.state_dict()); equal(a.opt_d.state_dict(),b.opt_d.state_dict())
    broken=copy.deepcopy(state);broken['particle_recipe']['loss']='hinge'
    with pytest.raises(ValueError,match='recipe differs'): restore_trainer(ReferenceTrainer(c),broken)


def test_legacy_config_keeps_legacy_formulation():
    assert ReferenceTrainer(resolve_config({})).recipe.critic_formulation=='ka2'


def test_conditioned_bispf_refused_before_training():
    c=config('bispf')
    c['components']['discriminator']['inputs']['condition']='batch.condition'
    with pytest.raises(ValueError,match='conditional-context'): ReferenceTrainer(c)
