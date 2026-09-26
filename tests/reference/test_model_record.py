"""Training records the per-layer network description the viewer's Model tab reads."""
import json
from pathlib import Path

import pytest

from hypergan.config import load_config, resolve_config, write_default
from hypergan.execution import train
from hypergan.model_description import MODEL_FILE, build_networks, describe_run

EXAMPLES = Path(__file__).resolve().parents[2] / 'examples'


def test_training_records_network_detail_for_its_configuration(tmp_path):
    config = write_default(tmp_path / 'config', device='cpu')
    run = tmp_path / 'run'
    train(config, run, steps=1, checkpoint_every=1)
    manifest = json.loads((run / 'manifest.json').read_text())
    recorded = json.loads((run / MODEL_FILE).read_text())
    assert recorded['config_sha256'] == manifest['config_sha256'] and recorded['origin'] == 'recorded'
    result = describe_run(manifest, recorded=recorded)
    for entry in result['networks']:
        graph = entry['graph']
        assert graph['status'] == 'built'
        subgraph = graph['subgraphs'][0]
        assert subgraph['node_count'] == len(subgraph['nodes']) > 0
        assert all(node['out'] for node in subgraph['nodes'])
        assert graph['parameters']['total'] == sum(node['params'] for node in subgraph['nodes']) > 0


def test_backfill_builds_per_layer_detail_on_the_meta_device():
    config = resolve_config({})
    recorded = build_networks(config, 'a' * 64)
    result = describe_run({'config': config, 'config_sha256': 'a' * 64}, recorded=json.loads(json.dumps(recorded)))
    for entry in result['networks']:
        graph = entry['graph']
        assert graph['status'] == 'built' and graph['origin'] == 'backfill'
        nodes = graph['subgraphs'][0]['nodes']
        assert nodes and all('out' in n and 'params' in n for n in nodes)
        assert sum(n['params'] for n in nodes) == graph['parameters']['total'] > 0


def test_backfill_captures_nodes_when_pretrained_weights_are_missing():
    config = load_config(EXAMPLES / 'dcgan-resnet-128.toml')
    weights = config['components']['discriminator']['args'].get('parameters', {}).get('weights_path')
    if weights and Path(weights).exists():
        pytest.skip('placeholder weights exist on this host')
    graph = build_networks(config)['components']['discriminator']
    assert graph['status'] == 'captured' and 'E_PRETRAINED' in graph['reason']
    assert graph['subgraphs'][0]['nodes'] and '/path/to' not in json.dumps(graph)
