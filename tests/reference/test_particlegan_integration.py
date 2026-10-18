"""Host contract for the released ParticleGAN preset."""
import copy

import pytest
import torch
from hndl.operators.pretrained import Pretrained
from particlegan import get_recipe

from hypergan.checkpoints import restore_trainer, trainer_state
from hypergan.config import config_values, load_config, resolve_config
from hypergan.training import ReferenceTrainer, initialize_fresh_parameters, particlegan_recipe


def config():
    return resolve_config({'defaults': 'particlegan',
        'components': {
            'generator': {'factory': 'hndl', 'args': {
                'source': 'linear(16)\nreshape(4,2,2)\nconv(2,kernel_size=1)\nflatten()\nlinear()',
                'input_shape': ['B', 4], 'output_shape': ['B', 2]}, 'inputs': {'x': 'latent'}},
            'discriminator': {'factory': 'hndl', 'args': {
                'source': 'linear(4)\nreshape(1,2,2)\nconv(2,kernel_size=1)\nflatten()\nlinear()',
                'input_shape': ['B', 2], 'output_shape': ['B', 1]}, 'inputs': {'x': 'candidate'}}},
        'prior': {'args': {'num_particles': 128, 'z_dim': 4}, 'initialization_seed': 2},
        'training': {'steps': 5, 'seed': 0, 'batch_size': 16, 'initialization': 'deterministic_orthogonal'}})


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    else:
        assert a == b


def test_complete_preset_factories_roles_references_and_exact_resume():
    torch.set_num_threads(1)
    trainer = ReferenceTrainer(config())
    expected = get_recipe()
    assert trainer.recipe.critic_formulation == expected.critic_formulation
    assert trainer.recipe.loss == expected.loss
    assert type(trainer.spread) is type(expected.make_prior_regularizer(prior=trainer.prior))
    reference = copy.deepcopy(trainer.prior.state_dict())
    assert [group['lr'] for group in trainer.opt_g.param_groups] == [
        expected.lr * (expected.prior_lr_mult if group['role'] == 'prior' else 1)
        for group in trainer.opt_g.param_groups]
    assert type(trainer.opt_d) is type(expected.make_critic_optimizer(copy.deepcopy(trainer.critic)))
    row, batch = trainer.update()
    assert torch.isfinite(torch.tensor(row['g_loss']))
    for name, buffer in trainer.prior.named_buffers():
        assert torch.equal(reference[name], buffer)
    saved = copy.deepcopy(trainer_state(trainer, batch))
    _, batch = trainer.update()
    wanted = copy.deepcopy(trainer_state(trainer, batch))
    resumed = ReferenceTrainer(config())
    restore_trainer(resumed, saved)
    _, batch = resumed.update()
    equal(wanted, trainer_state(resumed, batch))


def test_generated_image_project_uses_the_released_global_recipe(tmp_path):
    from PIL import Image
    from hypergan.recipe_generator import generate_recipe
    data = tmp_path / 'images'
    data.mkdir()
    Image.new('RGB', (32, 32)).save(data / 'one.png')
    path = tmp_path / 'model.toml'
    generate_recipe('test/images', data, output_path=path, device='cpu')
    resolved = load_config(path)
    recipe = particlegan_recipe(resolved)
    expected = get_recipe()
    assert recipe.name == expected.name
    assert (recipe.lr, recipe.prior_reg) == (expected.lr, expected.prior_reg)
    assert resolved['training']['initialization'] == 'deterministic_orthogonal'
    assert resolved['training']['seed'] == 0


def test_implementation_fingerprint_covers_the_dependency_package():
    import particlegan
    from pathlib import Path
    from hypergan.particlegan_integration import implementation_files
    paths = list(Path(particlegan.__file__).parent.rglob('*.py'))
    fingerprints = implementation_files()
    assert len(fingerprints) == len(paths)
    assert all(len(value) == 64 for value in fingerprints.values())


def test_importing_replicated_backend_cannot_change_the_native_recipe():
    import hypergan.training as training
    original = training.particlegan_recipe
    from hypergan import distributed_training
    assert training.particlegan_recipe is original
    assert training.particlegan_recipe(config()).to_dict() == original(config()).to_dict()
    from hypergan.execution_profiles import validate_replicated_recipe
    with pytest.raises(ValueError, match='require native execution'):
        validate_replicated_recipe(config())


def test_snapshot_supports_gaussian_prior_without_a_learned_table():
    raw = config_values(config())
    raw['prior'] = {'kind': 'gaussian', 'args': {'z_dim': 4}}
    raw['prior_regularizer']['weight'] = 0
    trainer = ReferenceTrainer(resolve_config(raw))
    row, _ = trainer.update()
    assert row['prior_loss'] == 0
    assert all(group['role'] != 'prior' for group in trainer.opt_g.param_groups)


def test_prior_optimizer_preserves_its_explicit_adam_epsilon():
    raw = config_values(config())
    raw['recipe'].update(eps=.03, prior_eps=.12)
    trainer = ReferenceTrainer(resolve_config(raw))
    assert all(group['eps'] == (.12 if group['role'] == 'prior' else .03)
               for group in trainer.opt_g.param_groups)
    row, _ = trainer.update()
    assert torch.isfinite(torch.tensor(row['g_loss']))


def test_inference_strict_load_preserves_recipe_prior_reference_buffers(tmp_path):
    from hypergan.artifacts import sample, save_bundle
    trainer = ReferenceTrainer(config())
    _, batch = trainer.update()
    save_bundle(tmp_path, trainer, batch)
    result = sample(tmp_path, count=4, seed=10000)
    assert result is not None


@pytest.mark.parametrize('damage', ['named_stream', 'global_rng', 'trainability'])
def test_invalid_late_checkpoint_fields_do_not_mutate_trainer(damage):
    trainer = ReferenceTrainer(config())
    _, batch = trainer.update()
    state = copy.deepcopy(trainer_state(trainer, batch))
    if damage == 'named_stream':
        state['streams']['data'] = torch.zeros(1, dtype=torch.uint8)
    elif damage == 'global_rng':
        state['rng']['torch'] = torch.zeros(1, dtype=torch.uint8)
    else:
        key = next(iter(state['trainable']['graph']))
        state['trainable']['graph'][key] = 'true'
    before = copy.deepcopy(trainer_state(trainer, batch))
    with pytest.raises(ValueError):
        restore_trainer(trainer, state)
    equal(before, trainer_state(trainer, batch))


@pytest.mark.parametrize('trainable', [True, False])
def test_public_initialization_preserves_loaded_trunk_and_effective_flags(trainable):
    # The provider/loading boundary is covered by HNDL's pretrained tests.
    # This fixture isolates HyperGAN's exclusion of already-loaded state.
    trunk = Pretrained.__new__(Pretrained)
    torch.nn.Module.__init__(trunk)
    trunk.model = torch.nn.Linear(4, 4)
    trunk.requires_grad_(trainable)
    head = torch.nn.Linear(4, 2)
    module = torch.nn.Sequential(trunk, head)
    graph = torch.nn.Module()
    graph.models = torch.nn.ModuleDict({'generator': module})
    before, head_before = copy.deepcopy(trunk.state_dict()), head.weight.clone()
    initialize_fresh_parameters(graph, 0)
    equal(before, trunk.state_dict())
    assert all(parameter.requires_grad == trainable for parameter in trunk.parameters())
    assert not torch.equal(head_before, head.weight)


@pytest.mark.parametrize('delta', [{'prior_proximal_strength': 0.1},
                                  {'constraint_geometry_mode': 'strict_progress'},
                                  {'conditioning': 'ucd', 'num_classes': 2},
                                  {'model': 'ddgan'}])
def test_unimplemented_recipe_mechanisms_fail_before_training(delta):
    raw = config()
    raw['recipe'].update(delta)
    with pytest.raises(ValueError, match='(does not|do not) implement'):
        particlegan_recipe(raw)
