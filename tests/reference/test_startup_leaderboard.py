"""The research table preserves evidence and refuses unmatched comparisons."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


_spec = importlib.util.spec_from_file_location(
    'startup_leaderboard', Path(__file__).resolve().parents[2] / 'research/startup_tuning/leaderboard.py')
board = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(board)


def report(mmd=(.5, .4, .3, .2), steps=(0, 8, 16, 32)):
    observations = []
    for index, (step, value) in enumerate(zip(steps, mmd)):
        image = {'absolute_above_0_99_fraction': .01 + index * .1,
                 'sample_diversity_rms': 1 / (index + 1),
                 'spatial_sample_diversity_rms': .5 / (index + 1)}
        features = {'protocol': 'frozen_dino_poly3', 'samples': 64, 'feature_width': 384,
                    'protected_state_sha256': 'frozen', 'dino_poly3_mmd2_unbiased': value,
                    'dino_real_feature_spread': .5, 'dino_fake_feature_spread': .25 / (index + 1),
                    'dino_feature_spread_ratio': .5 / (index + 1)}
        observations.append({'step': step, 'evolving_prior': {'output': image},
                             'fixed_latent': {'output': image}, 'frozen_features_evolving_prior': features})
    return {'kind': 'disposable-explicit-joint-rate-rollout', 'schema_version': 2,
            'config_sha256': 'config', 'original_config_fingerprint': 'resolved-config',
            'seed': 25002, 'device': 'cuda:0', 'source': {'hypergan_commit': 'revision'},
            'initial_parameters_sha256': 'initial-parameters',
            'g_lr': .0002, 'd_lr': .0002, 'prior_base_lrs': [.002],
            'requested_updates': steps[-1], 'observation_steps': list(steps),
            'budget': {'completed_native_training_updates': steps[-1]},
            'observations': observations, 'restored': True, 'source_config_unchanged': True,
            'protected_before_sha256': 'frozen', 'protected_after_sha256': 'frozen',
            'protected_after_restore_sha256': 'frozen', 'elapsed_seconds': 100.,
            'evaluation': {'protocol': 'joint-rate-fixed-bank', 'protocol_version': 2,
                           'bank_sha256': 'bank', 'prior_rng_sha256': 'prior-rng',
                           'measurement_rng_sha256': 'measurement-rng'}}


def write(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def test_baseline_first_matched_identity_and_no_composite_score(tmp_path):
    baseline = write(tmp_path, 'baseline.json', report())
    candidate = report((.5, .7, .45, .4))
    candidate['g_lr'] = .0001
    candidate['proposal'] = {'algorithm': 'example', 'resolved_changes': {'g_lr': .0001}}
    path = write(tmp_path, 'candidate.json', candidate)
    result = board.build_leaderboard(baseline, [path])
    rows = result['rows']
    assert [row['role'] for row in rows] == ['baseline', 'comparison']
    assert rows[1]['comparison_to_baseline']['comparable'] is True
    assert rows[0]['comparison_group'] == rows[1]['comparison_group']
    assert rows[0]['sustained_proxy_improvement']['first_observed_step'] == 8
    assert rows[1]['sustained_proxy_improvement']['first_observed_step'] == 16
    assert rows[1]['proposal'] == candidate['proposal']
    assert rows[1]['sustained_proxy_improvement']['first_observed_elapsed_seconds'] is None
    assert rows[1]['changes']['sample_diversity_rms_retention'] == .25
    assert not any(key in result for key in ('winner', 'score', 'ranking'))
    assert 'not training-only throughput' in board.markdown(result)


@pytest.mark.parametrize('values,step', [
    ((.5, .7, .6, .4), None),  # A lone final decrease is not sustained.
    ((.5, .4, .3, .5), None),  # Initial decrease followed by regression.
    ((.5, .5, .4, .3), 16),
    ((0., -.1, -.2, -.3), 8),  # Signed unbiased estimates remain valid.
])
def test_sustained_is_sign_only_and_requires_two_final_observations(tmp_path, values, step):
    path = write(tmp_path, 'baseline.json', report(values))
    row = board.build_leaderboard(path)['rows'][0]
    assert row['sustained_proxy_improvement']['first_observed_step'] == step
    assert row['changes']['dino_poly3_mmd2_unbiased_signed_change'] == pytest.approx(values[-1] - values[0])


@pytest.mark.parametrize('change', ['horizon', 'bank', 'seed', 'prior', 'initial_parameters', 'legacy'])
def test_unmatched_or_unverified_reports_never_share_comparison_group(tmp_path, change):
    baseline = write(tmp_path, 'baseline.json', report())
    other = report()
    if change == 'horizon':
        other = report(steps=(0, 8, 32, 128))
    elif change == 'bank':
        other['evaluation']['bank_sha256'] = 'different-bank'
    elif change == 'seed':
        other['seed'] += 1
    elif change == 'prior':
        other['prior_base_lrs'] = [.001]
    elif change == 'initial_parameters':
        other['initial_parameters_sha256'] = 'different-initial-parameters'
    else:
        other.pop('evaluation')
    candidate = write(tmp_path, 'candidate.json', other)
    rows = board.build_leaderboard(baseline, [candidate])['rows']
    assert rows[1]['comparison_to_baseline']['comparable'] is False
    assert rows[0]['comparison_group'] != rows[1]['comparison_group']


@pytest.mark.parametrize('fault', ['partial', 'protected', 'failure', 'missing_feature', 'zero_spread'])
def test_failed_incomplete_or_missing_measurements_are_not_success(tmp_path, fault):
    value = report()
    if fault == 'partial':
        value['budget']['completed_native_training_updates'] = 16
        value['observations'].pop()
    elif fault == 'protected':
        value['protected_after_sha256'] = 'changed-before-restore'
    elif fault == 'failure':
        value['failure'] = {'message': 'failed after final observation'}
    elif fault == 'missing_feature':
        value['observations'][1].pop('frozen_features_evolving_prior')
    else:
        value['observations'][0]['evolving_prior']['output']['sample_diversity_rms'] = 0.
    row = board.build_leaderboard(write(tmp_path, 'baseline.json', value))['rows'][0]
    if fault == 'zero_spread':
        assert row['changes']['sample_diversity_rms_retention'] is None
    else:
        assert row['sustained_proxy_improvement']['status'] == 'unavailable'


def test_wrapper_hash_verification_and_raw_sha_provenance(tmp_path):
    raw = write(tmp_path, 'raw.json', report())
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    wrapper = write(tmp_path, 'summary.json', {'raw_report': {'path': 'raw.json', 'sha256': digest}})
    result = board.build_leaderboard(wrapper, [raw])
    assert len(result['rows']) == 1
    assert result['rows'][0]['raw_report']['sha256'] == digest
    raw.write_text(raw.read_text() + '\n')
    with pytest.raises(ValueError, match='SHA256 does not match'):
        board.build_leaderboard(wrapper)


def test_cli_writes_json_and_markdown_without_overwriting_raw_input(tmp_path):
    raw = write(tmp_path, 'raw.json', report())
    before = raw.read_bytes()
    json_output, md_output = tmp_path / 'table.json', tmp_path / 'table.md'
    board.main(['--baseline', str(raw), '--json', str(json_output), '--markdown', str(md_output)])
    assert json.loads(json_output.read_text())['rows'][0]['role'] == 'baseline'
    assert 'Saturation initial → final' in md_output.read_text()
    with pytest.raises(ValueError, match='distinct'):
        board.main(['--baseline', str(raw), '--json', str(raw), '--markdown', str(md_output)])
    assert raw.read_bytes() == before


def test_nested_case_provenance_cost_and_observation_timing(tmp_path):
    value = report()
    value['proposal'] = {
        'case': 'source-control',
        'algorithm': {'name': 'source', 'proposal_seconds': .125,
                      'solution_path': '/research/solutions/source.json', 'solution_sha256': 'solution-hash',
                      'evidence': [{'path': '/research/evidence.json', 'sha256': 'evidence-hash', 'elapsed_seconds': 32.}]},
        'resolved': {'g_lr': .0002},
    }
    value['observations'][1]['timing_at_observation'] = {
        'elapsed_seconds': 12., 'training_update_seconds': 3., 'diagnostic_seconds': 7.}
    result = board.build_leaderboard(write(tmp_path, 'report.json', value))
    row = result['rows'][0]
    assert row['label'] == row['case'] == 'source-control'
    assert row['algorithm'] == value['proposal']['algorithm']
    assert row['resolved_changes'] == {'g_lr': .0002}
    assert row['elapsed_seconds'] == 100.
    assert row['proposal_seconds'] == .125
    assert row['sustained_proxy_improvement']['first_observed_elapsed_seconds'] == 12.
    rendered = board.markdown(result)
    assert '[source.json](</research/solutions/source.json>)' in rendered
    assert 'solution-hash' in rendered and 'evidence-hash' in rendered
    assert 'recorded creation seconds: 32' in rendered
    assert 'excludes computing it' in rendered
