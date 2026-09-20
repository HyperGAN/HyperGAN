"""Observation configuration and catalogs stay finite, immutable and torch-free."""
from copy import deepcopy
import json
import subprocess
import sys

import pytest

from hypergan.config import config_values, fingerprint, observation_fingerprint, resolve_config
from hypergan.metrics import metric_catalog, objective_id, publish_catalog, read_catalog, select_metrics


def test_default_none_and_individual_removal_do_not_change_numerical_identity(tmp_path):
    default = resolve_config({})
    none = resolve_config({'metrics': {'preset': 'none'}})
    removed = resolve_config({'metrics': {'disable': ['loss/total', 'loss/d_total'], 'every_steps': 3}})
    assert fingerprint(default) == fingerprint(none) == fingerprint(removed)
    assert observation_fingerprint(default) != observation_fingerprint(none)
    assert none['qualification'] == default['qualification']
    assert metric_catalog(none)['metrics'] == {}
    assert 'loss/total' not in metric_catalog(removed)['metrics']
    assert 'loss/d_total' not in metric_catalog(removed)['metrics']
    catalog = metric_catalog(default)
    for name in catalog['metrics']:
        config = resolve_config({'metrics': {'disable': [name]}})
        assert name not in metric_catalog(config)['metrics']
    revision = publish_catalog(tmp_path, default)
    before = (tmp_path / 'metrics' / f'catalog-{revision}.json').read_bytes()
    assert publish_catalog(tmp_path, default) == revision
    assert read_catalog(tmp_path, revision) == catalog
    assert (tmp_path / 'metrics' / f'catalog-{revision}.json').read_bytes() == before
    subprocess.run([sys.executable, '-I', '-c',
        'import sys; from hypergan.config import resolve_config; from hypergan.metrics import read_catalog; '
        'resolve_config({}); read_catalog(sys.argv[1],sys.argv[2]); assert "torch" not in sys.modules',
        str(tmp_path), revision], check=True)


@pytest.mark.parametrize('spec', [
    {'preset': 'fid'}, {'every_steps': 0}, {'every_steps': True}, {'every_steps': 1.5},
    {'disable': ['unknown']}, {'disable': ['loss/total', 'loss/total']}, {'disable': 'loss/total'},
    {'overrides': {'typo': {'enabled': False}}}, {'overrides': {'loss/total': {'enabled': 1}}},
    {'overrides': {'loss/total': {'label': 'custom'}}}, {'custom': {'fid': {'factory': 'm:FID'}}},
    {'custom': []}, {'disable': ['loss/total'], 'overrides': {'loss/total': {'enabled': True}}},
    {'unknown': 1},
])
def test_unsupported_or_ambiguous_metrics_fail_structurally(spec):
    with pytest.raises(ValueError):
        resolve_config({'metrics': spec})


def test_none_can_explicitly_select_one_and_total_is_independent_of_published_sources():
    config = resolve_config({'metrics': {'preset': 'none', 'overrides': {'loss/total': {'enabled': True}}}})
    values, statuses, mode = select_metrics(config, metric_catalog(config), {'d_loss': 2., 'g_loss': 3.}, 1, .1)
    assert values == {'loss/total': 5.} and not statuses and mode == 'sampled'
    config = resolve_config({'metrics': {'every_steps': 2}})
    assert select_metrics(config, metric_catalog(config), {}, 1, .1) == ({}, {}, 'cadence')


def test_zero_coefficients_and_lazy_schedule_do_not_fabricate_raw_values():
    config = resolve_config({'gradient_penalty': {'lazy_k': 2, 'coeff': 0}, 'adversarial': {'weight': 0}})
    row = {'d_adversarial_weighted': 0., 'g_adversarial_weighted': 0., 'd_adversarial': 3., 'g_adversarial': 4., 'gradient_penalty': 0.}
    catalog = metric_catalog(config)
    values, statuses, _ = select_metrics(config, catalog, row, 1, .1)
    assert values['loss/d_adversarial'] == 0
    assert values['loss/d_adversarial_raw'] == 3
    assert statuses['loss/gradient_penalty']['applied'] is False
    _, statuses, _ = select_metrics(config, catalog, row, 2, .1)
    assert statuses['loss/gradient_penalty']['applied'] is True
    assert statuses['loss/gradient_penalty']['effective_coefficient'] == 0
    assert statuses['loss/prior_regularizer']['status'] == 'unavailable'
    assert catalog['metrics']['loss/gradient_penalty']['raw_available'] is False
    with pytest.raises(ValueError, match='Nonfinite'):
        select_metrics(config, catalog, {'d_adversarial': float('nan')}, 2, .1)


def test_catalog_content_and_definition_integrity(tmp_path):
    config = resolve_config({})
    revision = publish_catalog(tmp_path, config)
    changed = resolve_config({'adversarial': {'weight': 2}})
    assert metric_catalog(config)['metrics']['loss/g_total']['definition_hash'] != metric_catalog(changed)['metrics']['loss/g_total']['definition_hash']
    with pytest.raises(ValueError, match='revision'):
        read_catalog(tmp_path, '../manifest')
    path = tmp_path / 'metrics' / f'catalog-{revision}.json'
    value = json.loads(path.read_text())
    value['metrics']['loss/total']['label'] = 'corrupted'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='revision'):
        publish_catalog(tmp_path, config)


def test_objective_names_survive_order_and_repeated_terms_require_ids():
    first = {'factory': 'mse', 'inputs': {'input': 'generated', 'target': 'batch.real'}}
    second = {'factory': 'l1', 'inputs': {'input': 'generated', 'target': 'batch.real'}}
    config = resolve_config({'objectives': [first, second]})
    names = [objective_id(term) for term in config['objectives']]
    reversed_config = resolve_config({'objectives': [second, first]})
    assert names == list(reversed([objective_id(term) for term in reversed_config['objectives']]))
    with pytest.raises(ValueError, match='unique'):
        resolve_config({'objectives': [first, first]})
    first['id'] = 'reconstruction'
    second['id'] = 'reconstruction_l1'
    config = resolve_config({'objectives': [first, second]})
    assert 'loss/objectives/reconstruction' in metric_catalog(config)['metrics']
    assert config_values(config)['objectives'][0]['id'] == 'reconstruction'


def evaluation_metric(**changes):
    spec = {'factory': 'uninstalled.metrics:Color', 'mode': 'snapshot',
            'inputs': {'generated': 'evaluation.generated', 'reference': 'evaluation.reference'},
            'evaluation': {'data': {'factory': 'gaussian_grid', 'args': {}}, 'sample_count': 8,
                           'batch_size': 4, 'seed': 5, 'device': 'cuda:1'}}
    spec['evaluation'].update(changes.pop('evaluation', {}))
    spec.update(changes)
    return spec


def test_manual_only_snapshot_metrics_warn_that_nothing_is_scheduled():
    config = resolve_config({'metrics': {'custom': {
        'fid_smoke': evaluation_metric(trigger='manual'),
        'fid50k_train': evaluation_metric(trigger='manual')}}})
    notice = [w for w in config['warnings'] if 'No automatic evaluation is scheduled' in w]
    assert len(notice) == 1 and 'fid50k_train, fid_smoke' in notice[0]
    assert 'hypergan evaluate' in notice[0] and '10000' in notice[0]
    # One scheduled metric means the run does publish evaluations; no notice.
    mixed = resolve_config({'metrics': {'custom': {
        'fid_smoke': evaluation_metric(trigger='manual'), 'fid50k_train': evaluation_metric()}}})
    assert not [w for w in mixed['warnings'] if 'No automatic evaluation' in w]
    # Disabling the only scheduled metric restores the notice.
    disabled = resolve_config({'metrics': {'disable': ['fid50k_train'], 'custom': {
        'fid_smoke': evaluation_metric(trigger='manual'), 'fid50k_train': evaluation_metric()}}})
    assert [w for w in disabled['warnings'] if 'No automatic evaluation' in w]
    assert not [w for w in resolve_config({})['warnings'] if 'No automatic evaluation' in w]


@pytest.mark.parametrize('training, evaluation, contends', [
    ('cuda', 'cuda', True), ('cuda:0', 'cuda', True), ('cuda', 'cuda:1', True),
    ('cuda:0', 'cuda:1', False), ('cuda', 'cpu', False), ('cpu', 'cuda:1', False),
    ('cpu', 'cpu', False)])
def test_interval_evaluation_warns_about_a_shared_training_device(training, evaluation, contends):
    config = resolve_config({'training': {'device': training},
                             'metrics': {'custom': {'fid': evaluation_metric(evaluation={'device': evaluation})}}})
    notice = [w for w in config['warnings'] if 'contention' in w]
    assert bool(notice) is contends
    if contends:
        assert 'fid' in notice[0] and training in notice[0]
    manual = resolve_config({'training': {'device': training}, 'metrics': {'custom': {
        'fid': evaluation_metric(trigger='manual', evaluation={'device': evaluation})}}})
    assert not [w for w in manual['warnings'] if 'contention' in w]
