"""Gate tests for the grouped first-generator contract. No training."""
import importlib.util
from pathlib import Path

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    'grouped_first_g', _ROOT / 'research/startup_tuning/algorithms/grouped_first_g.py')
decision = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(decision)
_probe_spec = importlib.util.spec_from_file_location(
    'grouped_first_g_probe', _ROOT / 'reports/grouped_first_g_probe.py')
probe = importlib.util.module_from_spec(_probe_spec)
_probe_spec.loader.exec_module(probe)


def _quadratic(s_a, s_b, *, a_a=-0.2, a_b=-0.1, k_a=0.8, k_b=0.4, k_ab=0.05, origin=1.0):
    return origin + a_a * s_a + a_b * s_b + 0.5 * k_a * s_a ** 2 + 0.5 * k_b * s_b ** 2 + k_ab * s_a * s_b


def _losses(**kwargs):
    return {name: _quadratic(*factors, **kwargs) for name, factors in {
        'origin': (0, 0), 'A_minus': (-1, 0), 'A_plus': (1, 0), 'B_minus': (0, -1),
        'B_plus': (0, 1), 'full_step': (1, 1), 'opposite_mix': (1, -1),
    }.items()}


def _parameters(scale):
    rows = []
    for path, response in zip(decision.GROUP_A, (0.4, 0.3, 0.2, 0.15)):
        rows.append({'path': path, 'projections': [response * scale] * 4})
    for index, response in enumerate((0.04, 0.03)):
        rows.append({'path': f'graph.models.generator.network.nodes.other_{index}.weight',
                     'projections': [response * scale] * 4})
    return rows


def _probe():
    group_b = [f'graph.models.generator.network.nodes.other_{index}.weight' for index in range(2)]
    bank = {'parameters': _parameters(1), 'losses': _losses(),
            'exact_slopes': {'A': -0.2, 'B': -0.1, 'missing_gradient': False},
            'output': {'rms_dA': 0.4, 'rms_dB': 0.2, 'rms_dAB': 0.7,
                       'rms_residual': 0.1, 'ms_cross': 0.02}}
    return {
        'group_a': list(decision.GROUP_A), 'group_b': group_b,
        'banks': [bank, {**bank, 'parameters': _parameters(1)}],
        'adam': {name: {'relative_rms': 0.0, 'contaminated_slope_fraction': 0.0, 'step': 1,
                        'weight_decay': 0, 'amsgrad': False, 'eps': 1e-8, 'lr': 2e-4}
                 for name in ('A', 'B')},
        'budget': {'native_updates': 1, 'loss_evaluations': 14,
                   'projection_backwards': 8, 'output_forwards': 4},
        'elapsed_seconds': 3.5,
        'bank_hashes': {'monitor': 'm', 'fitting_0': 'a', 'fitting_1': 'b'},
    }


def test_passing_quadratic_emits_the_smaller_shared_factor_without_global_rates():
    proposal = decision.decide(_probe())
    assert proposal['evidence']['decision'] == 'propose'
    assert set(proposal) == {'schema_version', 'layer_lr_multipliers', 'evidence'}
    assert [rule['pattern'] for rule in proposal['layer_lr_multipliers'][:4]] == list(decision.GROUP_A)
    assert len(proposal['layer_lr_multipliers']) == 6
    assert all(rule['multiplier'] == pytest.approx(0.25) for rule in proposal['layer_lr_multipliers'])
    assert proposal['evidence']['response_decomposition_forbidden'] is False
    assert proposal['evidence']['native_updates'] == 1
    assert proposal['evidence']['loss_evaluations'] == 14


def test_negative_curvature_abstains_before_the_unused_point():
    report = _probe()
    for bank in report['banks']:
        bank['losses'] = _losses(k_a=-0.8)
        bank['losses']['opposite_mix'] += 10
    proposal = decision.decide(report)
    assert proposal['evidence']['decision'] == 'abstain'
    assert proposal['evidence']['failed_gate'] == 'curvature'
    assert 'layer_lr_multipliers' not in proposal
    assert 'g_lr' not in proposal


def test_group_below_the_share_bar_does_not_fit_a_replacement():
    report = _probe()
    for bank in report['banks']:
        bank['parameters'][-1]['projections'] = [0.8] * 4
    proposal = decision.decide(report)
    assert proposal['evidence']['failed_gate'] == 'group_not_confirmed'
    assert decision.stencil_allowed(report) is False


def test_bank_gap_and_large_output_cross_term_follow_the_contract():
    report = _probe()
    report['banks'][1]['losses'] = _losses(a_a=-0.12)
    assert decision.decide(report)['evidence']['failed_gate'] == 'banks'
    report = _probe()
    for bank in report['banks']:
        bank['output']['rms_residual'] = 0.6
    proposal = decision.decide(report)
    assert proposal['evidence']['decision'] == 'propose'
    assert proposal['evidence']['response_decomposition_forbidden'] is True


def _exact_losses():
    # Separable factors are exactly 0.5 and 1, with a zero cross term.
    return {'origin': 0.0, 'A_minus': 1.0, 'A_plus': 0.0, 'B_minus': 1.5, 'B_plus': -0.5,
            'full_step': -0.5, 'opposite_mix': 1.5}


def test_exact_one_omits_that_group_and_both_ones_do_not_train():
    report = _probe()
    for bank in report['banks']:
        bank['losses'] = _exact_losses()
        bank['exact_slopes'] = {'A': -0.5, 'B': -1.0, 'missing_gradient': False}
    proposal = decision.decide(report)
    assert proposal['evidence']['decision'] == 'propose'
    assert {rule['pattern'] for rule in proposal['layer_lr_multipliers']} == set(decision.GROUP_A)
    assert proposal['layer_lr_multipliers'][0]['multiplier'] == 0.5
    report = _probe()
    for bank in report['banks']:
        bank['losses'] = {'origin': 0.0, 'A_minus': 1.5, 'A_plus': -0.5, 'B_minus': 1.5,
                          'B_plus': -0.5, 'full_step': -1.0, 'opposite_mix': 1.0}
        bank['exact_slopes'] = {'A': -1.0, 'B': -1.0, 'missing_gradient': False}
    assert decision.decide(report)['evidence']['failed_gate'] == 'no_change'


def test_missing_cost_is_null_and_blocks_a_proposal():
    report = _probe()
    del report['budget']['loss_evaluations']
    proposal = decision.decide(report)
    assert proposal['evidence']['decision'] == 'abstain'
    assert proposal['evidence']['failed_gate'] == 'incomplete_evidence'
    assert proposal['evidence']['loss_evaluations'] is None


def test_adam_record_accepts_a_proportional_step_and_rejects_a_floor():
    gradient = torch.tensor([1., -2., 1e-8])
    eps = 1e-8
    lr = 2e-4
    delta = -lr * gradient / (gradient.abs() + eps)
    record = probe.adam_group_record([(gradient, delta)], lr=lr, eps=eps, step=1, weight_decay=0, amsgrad=False)
    assert record['relative_rms'] == pytest.approx(0, abs=1e-6)
    assert record['contaminated_slope_fraction'] < 0.01
    shifted = delta.clone()
    shifted[0] += 1
    bad = probe.adam_group_record([(gradient, shifted)], lr=lr, eps=eps, step=1, weight_decay=0, amsgrad=False)
    assert bad['relative_rms'] > 1e-4
