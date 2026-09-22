#!/usr/bin/env python3
"""Baseline-first descriptive comparison of explicit joint-rate research reports.

Example:
  python reports/startup_leaderboard.py --baseline baseline.json candidate.json \
      --json leaderboard.json --markdown leaderboard.md

This script needs only Python's standard library. It never trains, changes a
configuration, assigns an aggregate score, or selects a winner. Compact report
wrappers may be supplied if their referenced raw report exists and SHA matches.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


CAVEATS = [
    'Frozen-DINO polynomial MMD is a noisy small-bank proxy, not Inception KID, statistical significance, or a quality/convergence certificate. Negative unbiased estimates are valid.',
    'Sustained means strictly below the initial MMD at every remaining observed checkpoint, with at least two such checkpoints. It says nothing about unobserved steps or behavior beyond the measured horizon.',
    'Image and feature diversity retention describe contraction or expansion; neither proves useful learning. Saturation is the output fraction with absolute value above 0.99.',
    'The online generator and evolving learned prior are measured together. Fixed-latent statistics are included separately to isolate generator motion.',
    'Total wall time includes setup, measurements and optional diagnostics. It is not training-only throughput or time to learn.',
    'Rollout elapsed time includes applying the proposal but excludes computing it. Proposal computation is recorded separately; producing prior evidence reports is an additional, generally unaccounted cost. Neither number is full research cost.',
    'Matching seed alone does not establish matching evaluation data. Missing protocol or evaluation-bank identity makes comparison unverified; different horizons remain separate.',
]


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _number(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def _ratio(after, before):
    return after / before if after is not None and before is not None and before > 0 else None


def _difference(after, before):
    return after - before if after is not None and before is not None else None


def _reject_constant(value):
    raise ValueError('Non-finite JSON number: ' + value)


def load_report(path):
    """Read exact bytes and verify a compact wrapper's referenced raw artifact."""
    supplied = Path(path).expanduser().resolve()
    data = supplied.read_bytes()
    report = json.loads(data, parse_constant=_reject_constant)
    if not isinstance(report, dict):
        raise ValueError('Research report must be a JSON object: ' + str(supplied))
    raw = supplied
    wrapper = None
    if isinstance(report.get('raw_report'), dict):
        reference = report['raw_report']
        raw = Path(reference['path']).expanduser()
        if not raw.is_absolute():
            raw = supplied.parent / raw
        raw = raw.resolve()
        wrapper = {'path': str(supplied), 'sha256': _sha(data)}
        data = raw.read_bytes()
        if _sha(data) != reference.get('sha256'):
            raise ValueError('Referenced raw report SHA256 does not match: ' + str(raw))
        report = json.loads(data, parse_constant=_reject_constant)
    if not isinstance(report, dict) or report.get('kind') != 'disposable-explicit-joint-rate-rollout':
        raise ValueError('Expected a raw disposable explicit joint-rate rollout report: ' + str(raw))
    provenance = {'path': str(raw), 'sha256': _sha(data)}
    if wrapper:
        provenance['wrapper'] = wrapper
    return report, provenance


def _observation(value):
    step = value.get('step')
    if type(step) is not int or step < 0:
        raise ValueError('Each observation requires a nonnegative integer step')
    current = value.get('evolving_prior', {}).get('output', {})
    fixed = value.get('fixed_latent', {}).get('output', {})
    feature = value.get('frozen_features_evolving_prior', {})
    affine = value.get('evolving_prior', {}).get('final_affine', {}).get('activation', {})
    return {
        'step': step,
        'elapsed_seconds': _number(value.get('elapsed_seconds', value.get('timing_at_observation', {}).get('elapsed_seconds'))),
        'timing_at_observation': value.get('timing_at_observation'),
        'saturation_fraction': _number(current.get('absolute_above_0_99_fraction')),
        'sample_diversity_rms': _number(current.get('sample_diversity_rms')),
        'spatial_sample_diversity_rms': _number(current.get('spatial_sample_diversity_rms')),
        'pooled_4x4_sample_diversity_rms': _number(current.get('pooled_4x4_sample_diversity_rms')),
        'mean_color_fraction_of_sample_variance': _number(current.get('mean_color_fraction_of_sample_variance')),
        'final_affine_rms': _number(affine.get('rms')),
        'pre_tanh_mean_derivative': _number(affine.get('tanh_response', {}).get('mean_derivative')),
        'fixed_latent_saturation_fraction': _number(fixed.get('absolute_above_0_99_fraction')),
        'fixed_latent_sample_diversity_rms': _number(fixed.get('sample_diversity_rms')),
        'dino_poly3_mmd2_unbiased': _number(feature.get('dino_poly3_mmd2_unbiased')),
        'dino_feature_mean_distance_rms': _number(feature.get('dino_feature_mean_distance_rms')),
        'dino_fake_feature_spread': _number(feature.get('dino_fake_feature_spread')),
        'dino_real_feature_spread': _number(feature.get('dino_real_feature_spread')),
        'dino_feature_spread_ratio': _number(feature.get('dino_feature_spread_ratio')),
        'real_score_mean': _number(feature.get('real_score_mean')),
        'fake_score_mean': _number(feature.get('fake_score_mean')),
        'matched_generator_adversarial_loss': _number(feature.get('matched_generator_adversarial_loss')),
    }


def _sustained(series, complete):
    values = [item['dino_poly3_mmd2_unbiased'] for item in series]
    result = {'status': 'unavailable', 'first_observed_step': None,
              'first_observed_elapsed_seconds': None,
              'definition': 'Every remaining observed MMD is strictly below step-zero MMD, with at least two remaining observations; sign-only, no significance claim'}
    if not complete:
        result['reason'] = 'A complete audited rollout through the requested horizon is required'
        return result
    if len(values) < 3 or any(value is None for value in values):
        result['reason'] = 'Requires step zero and at least two subsequent observations with finite feature MMD'
        return result
    for index in range(1, len(series) - 1):
        if all(value < values[0] for value in values[index:]):
            result.update(status='observed_sign_only', first_observed_step=series[index]['step'],
                          first_observed_elapsed_seconds=series[index]['elapsed_seconds'])
            return result
    result['status'] = 'not_observed'
    return result


def _identity(report, observations):
    """Require explicit bank/protocol identity; do not fabricate it for old rows."""
    evaluation = report.get('evaluation', {})
    first_feature = observations[0].get('frozen_features_evolving_prior', {}) if observations else {}
    identity = {
        'kind': report.get('kind'),
        'schema_version': report.get('schema_version'),
        'protocol': report.get('protocol') or ({'name': evaluation.get('protocol'), 'version': evaluation.get('protocol_version')}
                                             if isinstance(evaluation, dict) and evaluation.get('protocol') and evaluation.get('protocol_version') else None),
        'config_sha256': report.get('config_sha256'),
        'original_config_fingerprint': report.get('original_config_fingerprint'),
        'seed': report.get('seed'),
        'initial_parameters_sha256': report.get('initial_parameters_sha256'),
        'device': report.get('device'),
        'source': report.get('source'),
        'requested_updates': report.get('requested_updates'),
        'observation_steps': report.get('observation_steps'),
        'recorded_observation_steps': [value.get('step') for value in observations],
        'prior_base_lrs': report.get('prior_base_lrs'),
        'evaluation': evaluation,
        'feature_protocol': first_feature.get('protocol'),
        'feature_samples': first_feature.get('samples'),
        'feature_width': first_feature.get('feature_width'),
        'feature_extractor_state_sha256': first_feature.get('protected_state_sha256'),
    }
    missing = [name for name, value in identity.items() if value is None or value == '' or value == {} or (value == [] and name != 'prior_base_lrs')]
    if not isinstance(evaluation, dict) or not evaluation.get('bank_sha256'):
        missing.append('evaluation.bank_sha256')
    for key in ('measurement_rng_sha256', 'prior_rng_sha256'):
        if not isinstance(evaluation, dict) or not evaluation.get(key):
            missing.append('evaluation.' + key)
    feature_keys = ('protocol', 'samples', 'feature_width', 'protected_state_sha256')
    if any(any(value.get('frozen_features_evolving_prior', {}).get(key) != first_feature.get(key)
               for key in feature_keys) for value in observations):
        missing.append('consistent_feature_measurements')
    return identity, sorted(set(missing))


def summarize(report, raw_report, *, baseline=False):
    observations = sorted(report.get('observations', []), key=lambda value: value.get('step', -1))
    series = [_observation(value) for value in observations]
    if len({value['step'] for value in series}) != len(series):
        raise ValueError('Duplicate observation step in ' + raw_report['path'])
    requested = report.get('requested_updates')
    completed = report.get('budget', {}).get('completed_native_training_updates', 0)
    if type(requested) is not int or requested <= 0 or type(completed) is not int or completed < 0:
        raise ValueError('Requested/completed update counts must be nonnegative integers with a positive horizon')
    protected = [report.get(name) for name in (
        'protected_before_sha256', 'protected_after_sha256', 'protected_after_restore_sha256')]
    audit = {
        'restored': report.get('restored'),
        'source_config_unchanged': report.get('source_config_unchanged'),
        'protected_state_unchanged': bool(protected[0]) and all(value == protected[0] for value in protected),
        'protected_before_sha256': protected[0],
        'protected_after_sha256': protected[1],
        'protected_after_restore_sha256': protected[2],
    }
    audit['passed'] = (audit['restored'] is True and audit['source_config_unchanged'] is True
                       and audit['protected_state_unchanged'])
    complete = (completed == requested and bool(series) and series[0]['step'] == 0
                and series[-1]['step'] == requested and not report.get('failure') and audit['passed'])
    initial, final = (series[0], series[-1]) if series else ({}, {})
    initial_at_zero = initial.get('step') == 0
    changes = {}
    for metric in ('saturation_fraction', 'dino_poly3_mmd2_unbiased', 'dino_feature_mean_distance_rms'):
        changes[metric + '_signed_change'] = _difference(final.get(metric), initial.get(metric)) if initial_at_zero else None
    for metric in ('sample_diversity_rms', 'spatial_sample_diversity_rms',
                   'pooled_4x4_sample_diversity_rms', 'fixed_latent_sample_diversity_rms', 'dino_fake_feature_spread'):
        changes[metric + '_retention'] = _ratio(final.get(metric), initial.get(metric)) if initial_at_zero else None
    saturations = [value['saturation_fraction'] for value in series if value['saturation_fraction'] is not None]
    changes['observed_peak_saturation_fraction'] = max(saturations) if saturations else None
    identity, missing = _identity(report, observations)
    proposal = report.get('proposal') if isinstance(report.get('proposal'), dict) else {}
    case = report.get('case') or proposal.get('case')
    algorithm = report.get('algorithm') or proposal.get('algorithm')
    algorithm_metadata = algorithm if isinstance(algorithm, dict) else {}
    return {
        'label': str(case) if case else Path(raw_report['path']).stem,
        'role': 'baseline' if baseline else 'comparison',
        'raw_report': raw_report,
        'config': report.get('config'), 'config_sha256': report.get('config_sha256'),
        'original_config_fingerprint': report.get('original_config_fingerprint'),
        'source': report.get('source'), 'seed': report.get('seed'), 'device': report.get('device'),
        'rates': {'g_lr': report.get('g_lr'), 'd_lr': report.get('d_lr'),
                  'prior_base_lrs': report.get('prior_base_lrs'),
                  'initial_actual_lrs': report.get('per_step', [{}])[0].get('actual_lrs') if report.get('per_step') else None,
                  'final_actual_lrs': report.get('per_step', [{}])[-1].get('actual_lrs') if report.get('per_step') else None},
        'requested_updates': requested, 'completed_updates': completed,
        'complete_and_audited': complete, 'failure': report.get('failure'),
        'state_audit': audit, 'budget': report.get('budget'),
        'elapsed_seconds': _number(report.get('elapsed_seconds')),
        'proposal_seconds': _number(algorithm_metadata.get('proposal_seconds')),
        'cost_scope': 'Elapsed seconds cover the rollout including proposal application; proposal computation is separate and evidence-generation costs are not included',
        'timings': report.get('timings'),
        'timing_definition': report.get('timing_definition'),
        'proposal': report.get('proposal'),
        'case': case,
        'algorithm': algorithm,
        'resolved_changes': report.get('resolved_changes', proposal.get('resolved', proposal.get('resolved_changes'))),
        'initial': initial or None, 'final': final or None, 'changes': changes,
        'observations': series, 'sustained_proxy_improvement': _sustained(series, complete),
        'comparison_identity': identity, 'missing_comparison_identity': missing,
    }


def build_leaderboard(baseline_path, paths=()):
    rows, seen = [], set()
    for path in (baseline_path, *paths):
        report, raw = load_report(path)
        if raw['sha256'] in seen:
            continue
        seen.add(raw['sha256'])
        rows.append(summarize(report, raw, baseline=not rows))
    baseline = rows[0]
    for row in rows:
        differences = [name for name in row['comparison_identity']
                       if row['comparison_identity'][name] != baseline['comparison_identity'][name]]
        reasons = []
        if differences:
            reasons.append('Different ' + ', '.join(differences))
        if row['missing_comparison_identity'] or baseline['missing_comparison_identity']:
            reasons.append('Missing explicit identity; same seed does not verify the evaluation bank')
        if not row['complete_and_audited'] or not baseline['complete_and_audited']:
            reasons.append('Rollout incomplete or state audit did not pass')
        comparable = not reasons
        row['comparison_to_baseline'] = {
            'status': 'matched' if comparable else ('different' if differences else 'unverified'),
            'comparable': comparable, 'reasons': reasons,
        }
        # Unverified rows deliberately do not share a group merely because
        # both omitted the same metadata. Horizon is part of every identity.
        key = row['comparison_identity'] if not row['missing_comparison_identity'] else {
            'unverified_raw_sha256': row['raw_report']['sha256'], 'requested_updates': row['requested_updates']}
        row['comparison_group'] = _sha(_canonical(key))[:16]
    return {'schema_version': 1, 'kind': 'baseline-first-startup-leaderboard',
            'ordering': 'Designated baseline first, then input order; no ranking or aggregate score',
            'caveats': CAVEATS, 'rows': rows}


def _format(value, *, percent=False, signed=False):
    number = _number(value)
    if number is None:
        return 'n/a'
    if percent:
        return f'{100 * number:.2f}%'
    return format(number, '+.5g' if signed else '.5g')


def _cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def markdown(board):
    lines = ['# Startup measurements', '', board['ordering'] + '.', '',
             '| Run | G / D / prior LR | Updates | Saturation initial → final | Output diversity retention | DINO MMD Δ | Feature spread retention | Sustained proxy step | Rollout seconds | Proposal seconds | Compared with baseline |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in board['rows']:
        initial, final = row['initial'] or {}, row['final'] or {}
        rates, changes = row['rates'], row['changes']
        prior = ','.join(_format(value) for value in (rates['prior_base_lrs'] or [])) or 'n/a'
        proxy = row['sustained_proxy_improvement']
        proxy_text = str(proxy['first_observed_step']) if proxy['first_observed_step'] is not None else proxy['status'].replace('_', ' ')
        fields = [row['role'] + ': ' + row['label'],
                  _format(rates['g_lr']) + ' / ' + _format(rates['d_lr']) + ' / ' + prior,
                  f"{row['completed_updates']}/{row['requested_updates']}",
                  _format(initial.get('saturation_fraction'), percent=True) + ' → ' + _format(final.get('saturation_fraction'), percent=True),
                  _format(changes['sample_diversity_rms_retention']),
                  _format(changes['dino_poly3_mmd2_unbiased_signed_change'], signed=True),
                  _format(changes['dino_fake_feature_spread_retention']), proxy_text,
                  _format(row['elapsed_seconds']), _format(row['proposal_seconds']), row['comparison_to_baseline']['status']]
        lines.append('| ' + ' | '.join(_cell(value) for value in fields) + ' |')
    lines.extend(['', 'Negative MMD Δ means a lower measured proxy. Retention is final / initial; zero or missing denominators remain undefined. Proxy steps are observations, not a claim of useful learning.', ''])
    for row in board['rows']:
        lines.extend([f"- **{_cell(row['label'])}**: seed `{row['seed']}`, config SHA256 `{row['config_sha256']}`, group `{row['comparison_group']}`, audit {'passed' if row['state_audit']['passed'] else 'failed or unavailable'}.",
                      f"  Raw report: `{row['raw_report']['path']}`; SHA256 `{row['raw_report']['sha256']}`."])
        lines.append('  Budget: `' + json.dumps(row['budget'], sort_keys=True) + '`.')
        if row['timings']:
            lines.append('  Recorded timing components (seconds): `' + json.dumps(row['timings'], sort_keys=True) + '`.')
        algorithm = row['algorithm'] if isinstance(row['algorithm'], dict) else {}
        if algorithm:
            lines.append('  Algorithm: `' + _cell(algorithm.get('name', 'unspecified')) + '`; proposal computation: ' + _format(row['proposal_seconds']) + ' seconds.')
            if algorithm.get('solution_path'):
                path = str(algorithm['solution_path']).replace('>', '%3E')
                lines.append(f"  Solution: [{_cell(Path(path).name)}](<{path}>); SHA256 `{algorithm.get('solution_sha256', 'unavailable')}`.")
            if algorithm.get('path'):
                path = str(algorithm['path']).replace('>', '%3E')
                lines.append(f"  Algorithm source: [{_cell(Path(path).name)}](<{path}>); SHA256 `{algorithm.get('sha256', 'unavailable')}`.")
            for evidence in algorithm.get('evidence', []):
                path = str(evidence.get('path', 'unavailable')).replace('>', '%3E')
                cost = _format(evidence.get('elapsed_seconds'))
                lines.append(f"  Evidence: [{_cell(Path(path).name)}](<{path}>); SHA256 `{evidence.get('sha256', 'unavailable')}`; recorded creation seconds: {cost} (excluded from rollout/proposal times).")
        for reason in row['comparison_to_baseline']['reasons']:
            lines.append('  ' + reason + '.')
    lines.extend(['', *['- ' + item for item in board['caveats']], ''])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='*', help='Raw candidate reports, kept in input order')
    parser.add_argument('--baseline', required=True, help='Explicit baseline report, always shown first')
    parser.add_argument('--json', required=True, dest='json_output')
    parser.add_argument('--markdown', required=True, dest='markdown_output')
    args = parser.parse_args(argv)
    board = build_leaderboard(args.baseline, args.reports)
    destinations = [Path(args.json_output).expanduser().resolve(), Path(args.markdown_output).expanduser().resolve()]
    inputs = {Path(path).expanduser().resolve() for path in (args.baseline, *args.reports)}
    inputs.update(Path(row['raw_report']['path']) for row in board['rows'])
    if len(set(destinations)) != 2 or any(path in inputs for path in destinations):
        raise ValueError('Output paths must be distinct from each other and every input artifact')
    for path, content in zip(destinations, (json.dumps(board, indent=2, allow_nan=False) + '\n', markdown(board))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return board


if __name__ == '__main__':
    main()
