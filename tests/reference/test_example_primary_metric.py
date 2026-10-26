"""Model examples with several directed snapshot metrics name the one that defines "best"."""
import importlib
from pathlib import Path

import pytest

from hypergan.config import load_config
from hypergan.summary import primary_metric

EXAMPLES = Path(__file__).resolve().parents[2] / 'examples'


def _directions(config):
    """Catalog direction each configured snapshot metric declares, read from its plugin's describe()."""
    directions = {}
    for metric_id, spec in config['metrics'].get('custom', {}).items():
        if 'evaluation' not in spec:
            continue
        module, name = spec['factory'].split(':')
        cls = getattr(importlib.import_module(module), name)
        plugin = object.__new__(cls)  # describe() reads only its arguments; skip weight loading
        plugin.__dict__.update(spec.get('args') or {})
        directions[metric_id] = plugin.describe().get('direction')
    return directions


@pytest.mark.parametrize('example', ['logos-tiny-transformer-dinov3-projected-128.toml'])
def test_example_with_several_directed_metrics_has_one_primary_metric(example):
    config = load_config(EXAMPLES / example)
    directions = _directions(config)
    directed = [m for m, d in directions.items() if d in ('minimize', 'maximize')]
    assert len(directed) > 1, directions  # otherwise the catalog alone picks the metric
    records = [{'metric': m, 'direction': d, 'step': 1000, 'value': 1.0} for m, d in directions.items()]

    assert primary_metric(records, None) == (None, None)  # several directed metrics: no implicit pick
    metric, direction = primary_metric(records, config.get('checkpoints'))
    assert metric in directed
    assert direction == directions[metric]
