"""Training records the per-layer network description the viewer's Model tab reads."""
import json

from hypergan.config import write_default
from hypergan.execution import train
from hypergan.model_description import MODEL_FILE, describe_run


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
