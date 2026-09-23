"""A declared research case survives the real evaluator and artifact boundary."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from hypergan.config import load_config, write_default


@pytest.fixture
def harness(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / 'reports'))
    monkeypatch.syspath_prepend(str(root / 'research/startup_tuning'))
    spec = importlib.util.spec_from_file_location('research_benchmark_tests', root / 'research/startup_tuning/benchmark.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')
    return path


def make_manifest(tmp_path, solutions):
    config = write_default(tmp_path / 'config', device='cpu')
    config.write_text(config.read_text().replace('steps = 5', 'steps = 32')
                      .replace('num_particles = 20000', 'num_particles = 32'))
    references = []
    for solution in solutions:
        filename = solution['id'] + '.json'
        write_json(tmp_path / filename, {'schema_version': 1, **solution})
        references.append(filename)
    path = write_json(tmp_path / 'benchmark.json', {
        'schema_version': 1, 'name': 'tiny-native', 'config': str(config.relative_to(tmp_path)),
        'baseline': solutions[0]['id'], 'cases': references,
        'evaluation': {'steps': 2, 'observation_steps': [0, 1, 2],
                       'features': False, 'direction': False, 'crossed': False}})
    return path, config


def test_source_and_explicit_solution_run_through_identical_evaluator(harness, tmp_path):
    plan = {'g_lr': 1e-4, 'd_lr': 3e-4,
            'init_scales': [{'pattern': 'graph.models.generator.*weight', 'multiplier': .7}],
            'layer_lr_multipliers': [{'pattern': 'graph.models.generator.*weight', 'multiplier': .25}]}
    manifest_file, config_path = make_manifest(tmp_path, [
        {'id': 'source', 'algorithm': 'source'},
        {'id': 'explicit', 'algorithm': 'fixed', 'options': plan}])
    config_before = config_path.read_bytes()
    original = load_config(config_path)
    path, manifest = harness.load_manifest(manifest_file)
    reports = []
    for case in manifest['cases']:
        output = harness.run_case(path, manifest, case, tmp_path / 'results', 'cpu')
        report = json.loads(output.read_text())
        request = json.loads((output.parent / 'request.json').read_text())
        assert (output.parent / 'solution.json').read_bytes() == Path(case['_solution_path']).read_bytes()
        assert request['manifest_sha256'] == hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        assert request['algorithm']['solution_sha256'] == hashlib.sha256(Path(case['_solution_path']).read_bytes()).hexdigest()
        assert request['case'] == case['id'] == report['proposal']['case']
        assert report['status'] == 'complete' and report['completed_updates'] == 2
        assert report['restored'] and report['source_config_unchanged']
        assert [row['step'] for row in report['observations']] == [0, 1, 2]
        assert report['evaluation']['configured_training_horizon'] == 32
        assert report['evaluation']['diagnostics'] == {'features': False, 'direction': False, 'crossed': False}
        resolved = report['proposal']['resolved']
        assert resolved['prior_rates_unchanged'] and not resolved['pretrained_state_calibrated']
        assert resolved['effective_base_lrs'][0][1] == report['original_base_lrs'][0][1]
        for update in report['per_step']:
            for actual, base in zip(update['actual_lrs'], resolved['effective_base_lrs']):
                assert actual == pytest.approx([rate * update['lr_scale'] for rate in base])
        reports.append(report)
    baseline, candidate = reports
    assert config_path.read_bytes() == config_before
    assert baseline['g_lr'] == original['optimizer']['lr']
    assert baseline['d_lr'] == original['optimizer']['lr'] * original['optimizer']['d_lr_mult']
    assert candidate['g_lr'] == plan['g_lr'] and candidate['d_lr'] == plan['d_lr']
    assert baseline['initial_parameters_sha256'] == baseline['prepared_parameters_sha256']
    assert candidate['initial_parameters_sha256'] == baseline['initial_parameters_sha256']
    assert candidate['prepared_parameters_sha256'] != candidate['initial_parameters_sha256']
    assert baseline['evaluation']['bank_sha256'] == candidate['evaluation']['bank_sha256']
    assert baseline['evaluation']['measurement_rng_sha256'] == candidate['evaluation']['measurement_rng_sha256']
    generator = candidate['proposal']['resolved']['optimizers'][0]
    assert len(generator['group_base_lrs']) == len(baseline['original_base_lrs'][0]) + 1
    for parameter in generator['parameters']:
        multiplier = .25 if parameter['path'].endswith('weight') else 1.
        assert parameter['effective_lr'] == pytest.approx(plan['g_lr'] * multiplier)
    with pytest.raises(FileExistsError):
        harness.run_case(path, manifest, manifest['cases'][0], tmp_path / 'results', 'cpu')


def test_failed_proposal_preserves_solution_request_and_failure(harness, tmp_path):
    (tmp_path / 'broken.py').write_text('def propose(context):\n    raise ValueError("unresolved probe")\n')
    manifest_file, _ = make_manifest(tmp_path, [{'id': 'broken', 'algorithm': 'broken.py'}])
    path, manifest = harness.load_manifest(manifest_file)
    with pytest.raises(ValueError, match='unresolved probe'):
        harness.run_case(path, manifest, manifest['cases'][0], tmp_path / 'results', 'cpu')
    result = tmp_path / 'results/broken'
    assert (result / 'solution.json').exists() and (result / 'request.json').exists()
    report = json.loads((result / 'report.json').read_text())
    assert report['status'] == 'failed' and report['completed_updates'] == 0
    assert report['failure']['stage'] == 'proposal'
    assert 'restored' not in report


def test_custom_proposal_gets_copied_inputs_and_records_evidence_hashes(harness, tmp_path):
    evidence = write_json(tmp_path / 'evidence.json', {'recommended': .00015, 'description': 'one supplied measurement'})
    algorithm = tmp_path / 'custom.py'
    algorithm.write_text('''def propose(context):
    assert set(context) == {'config', 'options', 'evidence'}
    original = context['config']['optimizer']['lr']
    context['config']['optimizer']['lr'] = 999.
    return {'schema_version': 1, 'g_lr': context['evidence'][0]['recommended'],
            'd_lr': original * context['options']['d_multiplier']}
''')
    manifest_file, config_path = make_manifest(tmp_path, [{
        'id': 'custom', 'algorithm': 'custom.py', 'options': {'d_multiplier': .5},
        'evidence': ['evidence.json']}])
    path, manifest = harness.load_manifest(manifest_file)
    config = load_config(config_path)
    original_lr = config['optimizer']['lr']
    plan, provenance = harness.proposal_for(manifest['cases'][0], config, path)
    assert plan == {'schema_version': 1, 'g_lr': .00015, 'd_lr': original_lr * .5}
    assert config['optimizer']['lr'] == original_lr
    assert provenance['sha256'] == hashlib.sha256(algorithm.read_bytes()).hexdigest()
    assert provenance['evidence'] == [{'path': str(evidence), 'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest()}]
    assert provenance['proposal_seconds'] >= 0


@pytest.mark.parametrize('location', ['manifest', 'evaluation', 'solution'])
def test_seed_override_fields_are_rejected(harness, tmp_path, location):
    path, _ = make_manifest(tmp_path, [{'id': 'source', 'algorithm': 'source'}])
    target = tmp_path / 'source.json' if location == 'solution' else path
    document = json.loads(target.read_text())
    container = document['evaluation'] if location == 'evaluation' else document
    container['seed'] = 123
    write_json(target, document)
    with pytest.raises(ValueError, match='fields'):
        harness.load_manifest(path)
