"""The viewer's model description: torch-free, bounded, redacted, for every example."""
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from hypergan.config import load_config, resolve_config
from hypergan.metrics import objective_id
from hypergan.model_description import (MAX_NODES, _merge_recorded, describe_config, describe_run, redact,
                                        redact_text)

EXAMPLES = sorted(Path(__file__).resolve().parents[2].joinpath('examples').glob('*.toml'))
LOCAL_PATH = re.compile(r'"(?:/home|/mnt|/root|/Users|~/)')


def test_examples_exist():
    assert len(EXAMPLES) >= 20


@pytest.mark.parametrize('path', EXAMPLES, ids=lambda p: p.stem)
def test_every_example_describes(path):
    config = load_config(path)
    result = describe_run({'config': config, 'config_sha256': 'c' * 64, 'run_id': 'run'})
    text = json.dumps(result, allow_nan=False)
    assert not LOCAL_PATH.search(text), 'absolute local path leaked'
    assert len(text) < 256 * 1024
    assert result['formulation']['family'] == 'k3p'
    assert result['formulation']['parameters']['coeff'] == config['gradient_penalty']['coeff']
    names = [entry['name'] for entry in result['networks']]
    assert names == list(config['components'])
    assert {'generator', 'discriminator'} <= set(names)
    roles = {entry['name']: entry['role'] for entry in result['networks']}
    assert roles['generator'] == 'generator' and roles['discriminator'] == 'critic'
    for entry in result['networks']:
        assert entry['graph']['status'] == 'unavailable' and entry['graph']['reason']
        spec = config['components'][entry['name']]
        if spec.get('factory') == 'hndl':
            assert entry['source']['text'].strip()
            assert entry['input_shape'] == spec['args']['input_shape']
    # Every loss term names its series, or says why it has none.
    for side in ('discriminator', 'generator'):
        for term in result['losses'][side]:
            assert term['metric'] or term['metric_note']
    objectives = {term['metric'] for term in result['losses']['generator'] if term['kind'] not in (
        'rpgan_g', 'particle_spread')}
    assert objectives == {'loss/objectives/' + objective_id(t) for t in config['objectives']}
    assert result['optimizers']['critic']['lr'] == pytest.approx(
        config['optimizer']['lr'] * config['optimizer']['d_lr_mult'])


def test_multi_critic_terms_are_listed_without_a_per_term_series():
    config = load_config(next(p for p in EXAMPLES if p.stem == 'color-words'))
    result = describe_config(config)
    critics = [t['critic'] for t in result['losses']['discriminator'] if t['kind'] == 'rpgan_d']
    assert critics == ['discriminator'] + [t['component'] for t in config['adversarial_terms']]
    assert all(t['metric'] is None and 'summed' in t['metric_note'] for t in result['losses']['discriminator'])
    assert set(result['optimizers']['critic']['components']) == set(critics)
    assert any(e['role'] == 'alias' and e['reuse_of'] for e in result['networks'])


def test_encoder_role_and_objective_edges():
    config = load_config(next(p for p in EXAMPLES if p.stem == 'cifar-transgan'))
    result = describe_config(config)
    roles = {entry['name']: entry['role'] for entry in result['networks']}
    assert roles['encoder'] == 'encoder' and roles['reconstruction'] == 'alias'
    assert {'from': 'components.generator', 'to': 'components.reconstruction', 'port': 'weights',
            'path': 'reuse'} in result['edges']
    assert any(e['to'].startswith('objective.') for e in result['edges'])


def test_default_configuration_and_packaged_source_file():
    result = describe_config(resolve_config({}))
    generator = result['networks'][0]
    assert generator['source']['file'] == 'hypergan/networks/reference.hndl'
    assert result['prior']['kind'] == 'particles'
    assert result['provenance']['defaults_source'] == 'unknown'


def test_legacy_configuration_is_shown_not_rederived():
    config = json.loads(json.dumps(resolve_config({})))
    config['adversarial'].update(loss_type='logistic', mode='rp')
    config['gradient_penalty'].update(arm='b_cap', norm='l2', target_anneal=0.5)
    for key in ('anchor_weight', 'anchor_decay'):
        del config['gradient_penalty'][key]
    result = describe_run({'config': config, 'source': {'particlegan_distribution_version': '0.5.0'}})
    formulation = result['formulation']
    assert formulation['family'] == 'legacy' and formulation['particlegan'] == '0.5.0'
    assert formulation['parameters']['arm'] == 'b_cap' and 'penalty' not in formulation['equations']
    assert result['losses']['discriminator'][1]['kind'] == 'penalty_b_cap'


def test_missing_configuration_is_not_found():
    with pytest.raises(FileNotFoundError):
        describe_run({'run_id': 'run'})


def test_local_paths_are_redacted_everywhere():
    assert redact({'root': '/home/me/data', 'rel': 'data/x', 'w': ['~/w.pth', 'C:\\m\\x.pt']}) == {
        'root': '…/data', 'rel': 'data/x', 'w': ['…/w.pth', '…/x.pt']}
    assert redact_text('E: /home/me/r18.pth does not exist') == 'E: …/r18.pth does not exist'
    config = json.loads(json.dumps(resolve_config({})))
    config['data']['args']['root'] = '/home/someone/private/data'
    config['components']['generator']['args']['source'] += '\n# weights "/mnt/private/x.pth"\n'
    manifest = {'config': config, 'warnings': ['cache at /home/someone/cache/file.bin']}
    text = json.dumps(describe_run(manifest), ensure_ascii=False)
    assert 'private' not in text and 'someone' not in text and '…/data' in text


def test_recorded_detail_merges_only_for_the_same_configuration():
    config = resolve_config({})
    node = {'id': 'n0', 'op': 'linear', 'args': {'weights': '/home/x/w.pth'}, 'params': 12}
    recorded = {'config_sha256': 'a' * 64, 'components': {'generator': {
        'status': 'built', 'unknown': 'dropped', 'parameters': {'total': 12, 'trainable': 12, 'frozen': 0},
        'subgraphs': [{'module_path': 'network', 'node_count': 1, 'nodes': [node] * (MAX_NODES + 5)}]}}}
    merged = describe_run({'config': config, 'config_sha256': 'a' * 64}, recorded=recorded)
    graph = merged['networks'][0]['graph']
    assert graph['status'] == 'built' and 'unknown' not in graph and graph['origin'] == 'recorded'
    assert len(graph['subgraphs'][0]['nodes']) == MAX_NODES
    assert graph['subgraphs'][0]['nodes'][0]['args']['weights'] == '…/w.pth'
    assert merged['networks'][1]['graph']['status'] == 'unavailable'
    stale = _merge_recorded(describe_config(config), recorded, 'b' * 64)
    assert all('different configuration' in e['graph']['reason'] for e in stale['networks'])


def test_catalog_marks_series_the_run_does_not_publish():
    result = describe_run({'config': resolve_config({})}, catalog={'metrics': {'loss/d_adversarial': {}}})
    by_id = {t['id']: t for t in result['losses']['generator']}
    assert by_id['adversarial']['metric_note'] == "not in this run's metric catalog"


def test_description_never_imports_torch(tmp_path):
    code = ("import sys, json; from hypergan.config import load_config; "
            "from hypergan.model_description import describe_run; "
            "[json.dumps(describe_run({'config': load_config(p)})) for p in sys.argv[1:]]; "
            "assert 'torch' not in sys.modules and 'hndl' not in sys.modules and 'particlegan' not in sys.modules")
    subprocess.run([sys.executable, '-c', code, *map(str, EXAMPLES)], cwd=tmp_path, check=True)


def test_backfill_builds_per_layer_detail_on_the_meta_device():
    from hypergan.model_description import build_networks
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
    from hypergan.model_description import build_networks
    config = load_config(next(p for p in EXAMPLES if p.stem == 'dcgan-resnet-128'))
    weights = config['components']['discriminator']['args'].get('parameters', {}).get('weights_path')
    if weights and Path(weights).exists():
        pytest.skip('placeholder weights exist on this host')
    graph = build_networks(config)['components']['discriminator']
    assert graph['status'] == 'captured' and 'E_PRETRAINED' in graph['reason']
    assert graph['subgraphs'][0]['nodes'] and '/path/to' not in json.dumps(graph)
