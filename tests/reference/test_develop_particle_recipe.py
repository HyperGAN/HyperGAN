"""Develop recipe transfer, actual updates and exact resume through HyperGAN."""
import copy
import pytest
import torch
from hypergan.config import resolve_config, config_values, load_config, fingerprint
from hypergan.training import ReferenceTrainer
from hypergan.checkpoints import trainer_state, restore_trainer
import hypergan.distributed_training  # import order must not override the native recipe factory


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


@pytest.mark.parametrize('trainer',['bcap','bispf','ka2','k3p'])
def test_recipe_selector_survives_every_configuration_boundary(trainer):
    c=config(trainer)
    roundtrip=load_config(config_values(c))
    assert roundtrip['particlegan_trainer']==trainer
    assert fingerprint(roundtrip)==fingerprint(c)
    assert ReferenceTrainer(roundtrip).recipe.critic_formulation==trainer
    changed=copy.deepcopy(c);changed['particlegan_trainer']='ka2' if trainer!='ka2' else 'bcap'
    assert fingerprint(changed)!=fingerprint(c)


def test_public_training_checkpoint_and_resume_keep_selected_engine(tmp_path):
    from hypergan.execution import train,resume
    from hypergan.checkpoints import read_checkpoint
    c=config('bcap');c['training']['steps']=3
    first=train(c,tmp_path/'run',stop_after_steps=1,checkpoint_every=1)
    assert first['config']['particlegan_trainer']=='bcap'
    payload=read_checkpoint(tmp_path/'run')[2]
    assert payload['particle_recipe']['critic_formulation']=='bcap'
    assert 'direction_blend' in payload['optimizers'][0]
    last=resume(tmp_path/'run')
    assert last['status']=='complete' and last['steps']==3
    assert last['config']['particlegan_trainer']=='bcap'
    with pytest.raises(ValueError,match='configuration|differs|compatible'):
        other=copy.deepcopy(c);other['particlegan_trainer']='ka2'
        resume(tmp_path/'run',config_path=other)


def test_public_training_refuses_conditional_bispf_before_run_creation(tmp_path):
    from hypergan.execution import train
    from pathlib import Path
    c=config('bispf')
    c['components']['discriminator']['inputs']['condition']='batch.condition'
    run=tmp_path/'must-not-exist'
    with pytest.raises(ValueError,match='conditional-context'): train(c,run)
    assert not run.exists()


def test_replicated_import_never_replaces_the_native_recipe_factory():
    import importlib
    import hypergan.training as native
    import hypergan.distributed_training as replicated
    factory=native.particlegan_recipe
    importlib.reload(replicated)
    assert native.particlegan_recipe is factory
    for trainer in ('bcap','bispf'):
        assert ReferenceTrainer(config(trainer)).recipe.critic_formulation==trainer


@pytest.mark.parametrize('trainer',['bcap','bispf'])
def test_replicated_public_execution_refuses_native_only_presets_before_launch(tmp_path,trainer):
    from hypergan.execution import train
    c=config(trainer)
    profile={'schema_version':1,'execution':{'name':'cpu-replicated-gloo'}}
    run=tmp_path/'not-launched'
    with pytest.raises(ValueError,match='require native'):
        train(c,run,profile=profile)
    assert not run.exists()


def test_module_cli_import_is_available_in_the_installed_package():
    from hypergan.cli import _parser
    assert _parser().parse_args(['train','config.toml','--no-server']).command=='train'


def test_lightweight_recipe_defaults_match_exact_pinned_source():
    import dataclasses
    from particlegan.recipes import Recipe
    from hypergan.particle_integration import PINNED_RECIPE_DEFAULTS
    assert PINNED_RECIPE_DEFAULTS == {f.name:f.default for f in dataclasses.fields(Recipe)}
