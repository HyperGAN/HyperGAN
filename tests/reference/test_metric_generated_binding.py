"""CPU snapshot output overrides without a training run or worker process."""
import copy
import hashlib

import pytest
import torch

from hypergan.artifacts import bundle_state
from hypergan.config import resolve_config
from hypergan.metric_evaluation_worker import evaluate_snapshot
from hypergan.metric_plugins import _factory
from hypergan.training import ReferenceTrainer


class MeanSquaredDistance:
    def describe(self):
        return {'kind': 'scalar', 'direction': 'minimize'}

    def evaluate(self, *, batches, context):
        total, count = 0., 0
        for batch in batches:
            errors = (batch['generated'] - batch['reference']).square()
            total += float(errors.sum())
            count += errors.numel()
        return total / count


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def recipe():
    config = resolve_config({})
    # Use the normal 2D reference generator; a reconstruction chain copies
    # reference inputs exactly, making output-selection mistakes measurable.
    config['training']['device'] = 'cpu'
    config['prior']['args']['num_particles'] = 16
    config['components']['reference_copy'] = {
        'factory': 'identity', 'inputs': {'input': 'batch.real'}, 'trainable': False}
    config['components']['reconstruction'] = {
        'factory': 'identity', 'inputs': {'input': 'components.reference_copy'}, 'trainable': False}
    config['objectives'] = [{'factory': 'mse', 'inputs': {
        'input': 'generated', 'target': 'components.reconstruction'}}]
    config['metrics']['custom'] = {'distance': {
        'factory': __name__ + ':MeanSquaredDistance', 'mode': 'snapshot', 'trigger': 'manual',
        'inputs': {'generated': 'evaluation.generated', 'reference': 'evaluation.reference'},
        'evaluation': {'device': 'cpu', 'sample_count': 7, 'batch_size': 3, 'seed': 83,
                       'data': {'factory': 'gaussian_grid', 'args': {'side': 3, 'noise': .01}}}}}
    from hypergan.config import config_values
    return config_values(config)


def snapshot(tmp_path, config):
    trainer = ReferenceTrainer(resolve_config(config))
    trainer.artifact_identity = {'run_id': 'fixture', 'attempt_id': 'attempt'}
    bundle = bundle_state(trainer, {'real': torch.zeros(2, 2)})
    path = tmp_path / 'snapshot.pt'
    torch.save(bundle, path)
    return trainer, bundle, path


def evaluate(path, spec, identity):
    expected = _factory(spec['factory'], spec['args'])[1]
    return evaluate_snapshot(spec, expected, str(path), hashlib.sha256(path.read_bytes()).hexdigest(), identity)


def test_snapshot_override_chooses_random_output_and_records_protocol(tmp_path):
    config = recipe()
    config['sampling']['generated'] = 'components.reconstruction'
    trainer, bundle, path = snapshot(tmp_path, config)
    spec = trainer.config['metrics']['custom']['distance']
    conditional = evaluate(path, spec, trainer.artifact_identity)
    assert conditional['value'] == 0
    assert conditional['protocol']['generated_binding'] == 'components.reconstruction'
    random_spec = copy.deepcopy(spec)
    random_spec['evaluation']['generated'] = 'generated'
    random_result = evaluate(path, random_spec, trainer.artifact_identity)
    assert random_result['value'] > 0
    assert random_result['protocol']['generated_binding'] == 'generated'
    assert random_result['protocol']['evaluation']['generated'] == 'generated'
    assert trainer.config['sampling']['generated'] == 'components.reconstruction'


def test_snapshot_retains_metric_component_dependencies_without_sampler_inputs(tmp_path):
    config = recipe()
    trainer = ReferenceTrainer(resolve_config(config))
    assert set(bundle_state(trainer, {})['components']) == {'generator'}
    config['metrics']['custom']['distance']['evaluation']['generated'] = 'components.reconstruction'
    trainer, bundle, path = snapshot(tmp_path, config)
    assert set(bundle['components']) == {'generator', 'reference_copy', 'reconstruction'}
    assert bundle['example_inputs'] == {}  # Evaluation supplies its own data.
    result = evaluate(path, trainer.config['metrics']['custom']['distance'], trainer.artifact_identity)
    assert result['value'] == 0
    assert result['protocol']['generated_binding'] == 'components.reconstruction'


def test_new_override_on_old_snapshot_fails_clearly(tmp_path):
    trainer, _, path = snapshot(tmp_path, recipe())
    spec = copy.deepcopy(trainer.config['metrics']['custom']['distance'])
    spec['evaluation']['generated'] = 'components.reconstruction'
    with pytest.raises(ValueError, match='Snapshot does not contain evaluation.generated component'):
        evaluate(path, spec, trainer.artifact_identity)


@pytest.mark.parametrize('binding', [None, '', 'latent', 'batch.real', 'candidate',
                                     'components.', 'components.missing', 'components.discriminator'])
def test_invalid_or_unavailable_evaluation_output_rejected(binding):
    config = recipe()
    config['metrics']['custom']['distance']['evaluation']['generated'] = binding
    with pytest.raises(ValueError, match='evaluation.generated|component binding'):
        resolve_config(config)
