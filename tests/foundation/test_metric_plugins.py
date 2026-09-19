"""Custom factories are structurally validated without importing their code."""
from copy import deepcopy
import subprocess
import sys

import pytest

from hypergan.config import fingerprint, resolve_config


def scalar():
    return {'factory': 'uninstalled.metrics:Ratio', 'inputs': {'numerator': 'update.g_loss', 'denominator': 'update.d_loss'}}


def snapshot():
    return {'factory': 'uninstalled.metrics:Color', 'mode': 'snapshot', 'trigger': 'manual',
            'inputs': {'generated': 'evaluation.generated', 'reference': 'evaluation.reference'},
            'evaluation': {'data': {'factory': 'gaussian_grid', 'args': {}},
                           'sample_count': 8, 'batch_size': 4, 'seed': 5}}


def test_custom_structural_configuration_never_imports_factory():
    config = resolve_config({'metrics': {'custom': {'ratio': scalar(), 'color': snapshot()}}})
    assert 'uninstalled.metrics' not in sys.modules
    assert config['metrics']['custom']['color']['evaluation']['device'] == 'cuda'
    assert fingerprint(config) == fingerprint(resolve_config({}))
    code = 'import sys; from hypergan.config import resolve_config; resolve_config({"metrics":{"custom":{"custom":{"factory":"nonexistent.module:C","inputs":{"value":"update.g_loss"}}}}}); assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('changes', [
    {'factory': 'missingcolon'}, {'inputs': {'tensor': 'trainer.latent'}}, {'inputs': {}},
    {'inputs': {'context': 'update.g_loss'}}, {'mode': 'stream-stateful'}, {'every_steps': 0},
    {'timeout': float('inf')}, {'on_error': 'ignore'}, {'args': {'bad': float('nan')}},
    {'unknown': True}, {'evaluation': {}},
])
def test_invalid_scalar_modes_and_bindings_fail(changes):
    value = scalar()
    value.update(changes)
    with pytest.raises(ValueError):
        resolve_config({'metrics': {'custom': {'ratio': value}}})


def test_snapshot_scheduling_is_explicitly_unsupported_and_data_is_required():
    for changes in ({'trigger': 'scheduled'}, {'every_steps': 2}, {'evaluation': {}}, {'inputs': {'x': 'update.g_loss'}}):
        spec = snapshot()
        spec.update(changes)
        with pytest.raises(ValueError):
            resolve_config({'metrics': {'custom': {'color': spec}}})


def test_custom_cannot_replace_builtin_but_can_be_disabled():
    with pytest.raises(ValueError, match='replace'):
        resolve_config({'metrics': {'custom': {'loss/total': scalar()}}})
    config = resolve_config({'metrics': {'preset': 'none', 'custom': {'ratio': scalar()}, 'disable': ['ratio']}})
    from hypergan.metric_plugins import prepare_custom
    prepare_custom(config)
    assert config['_metric_runtime'] == {}


def test_manual_only_catalog_does_not_claim_sampled_training_values():
    from hypergan.metrics import metric_catalog, select_metrics
    config=resolve_config({'metrics':{'preset':'none','custom':{'color':snapshot()}}})
    config['_metric_runtime']={'color':{'descriptor':{'kind':'scalar'},'factory_sources':{'module':'a'*64},'protocol':'hypergan-metric/v1'}}
    assert select_metrics(config,metric_catalog(config),{},1,.1)==({}, {}, 'disabled')
