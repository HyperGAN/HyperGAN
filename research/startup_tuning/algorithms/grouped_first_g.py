"""Apply the grouped first-generator contract to one saved probe.

The function does not train, draw a bank, or replace a failed gate with a
smaller step. Abstention returns no learning-rate fields.
"""
import math


GROUP_A = (
    'graph.models.generator.network.nodes.n_stage8_block1_ffn.down.weight',
    'graph.models.generator.network.nodes.n_stage8_block0_ffn.down.weight',
    'graph.models.generator.network.nodes.n_stage16_block0_ffn.down.weight',
    'graph.models.generator.network.nodes.n_stage16_block1_ffn.down.weight',
)
_FORMULA = ('L = L00 + aA*sA + aB*sB + (kA/2)*sA^2 + (kB/2)*sB^2 + kAB*sA*sB; '
            'symmetric a=(L(+1)-L(-1))/2; k=L(+1)+L(-1)-2*L00; '
            'kAB=L(1,1)-L00-aA-aB-kA/2-kB/2; s*=-a/k only inside (0, 1]')
_LOSS_KEYS = ('origin', 'A_minus', 'A_plus', 'B_minus', 'B_plus', 'full_step', 'opposite_mix')
_GATES = ('curvature', 'hessian', 'slopes', 'separable_factors', 'banks',
          'cross_term', 'unused_point', 'adam_step_not_proportional')


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _unit_interval(value):
    return _finite(value) and 0 < value <= 1


def _squared(projections):
    if not isinstance(projections, list) or len(projections) != 4:
        return 'missing'
    if any(not _finite(value) for value in projections):
        return 'nonfinite'
    return sum(value * value for value in projections) / 4


def _cost(probe):
    budget = probe.get('budget') if isinstance(probe.get('budget'), dict) else {}
    hashes = probe.get('bank_hashes') if isinstance(probe.get('bank_hashes'), dict) else {}
    elapsed = probe.get('elapsed_seconds')
    copied = {}
    for name in ('native_updates', 'loss_evaluations', 'projection_backwards', 'output_forwards'):
        value = budget.get(name)
        copied[name] = value if type(value) is int and value >= 0 else None
    copied['elapsed_seconds'] = elapsed if _finite(elapsed) and elapsed >= 0 else None
    copied['bank_hashes'] = {name: hashes.get(name) if isinstance(hashes.get(name), str) and hashes.get(name) else None
                             for name in ('monitor', 'fitting_0', 'fitting_1')}
    return copied


def _abstain(probe, gate, **detail):
    evidence = {'decision': 'abstain', 'failed_gate': gate, 'formula': _FORMULA, **_cost(probe), **detail}
    return {'schema_version': 1, 'evidence': evidence}


def _rank_bank(bank, group_a):
    parameters = bank.get('parameters') if isinstance(bank, dict) else None
    if not isinstance(parameters, list) or not parameters:
        return 'missing', None
    rows = []
    for item in parameters:
        if not isinstance(item, dict) or not isinstance(item.get('path'), str):
            return 'missing', None
        squared = _squared(item.get('projections'))
        if squared == 'nonfinite':
            return 'nonfinite', None
        if squared == 'missing':
            return 'missing', None
        rows.append((squared, item['path']))
    if len({path for _, path in rows}) != len(rows):
        return 'group_identity', None
    ranked = sorted(rows, key=lambda item: (-item[0], item[1]))
    total = sum(value for value, _ in rows)
    if not total > 0:
        return 'group_not_confirmed', {'share': None, 'top4': [path for _, path in ranked[:4]]}
    share = sum(value for value, path in rows if path in group_a) / total
    return None, {'share': share, 'top4': [path for _, path in ranked[:4]],
                  'ranked': [{'path': path, 'estimated_squared_response': value} for value, path in ranked]}


def _coefficients(losses):
    if not isinstance(losses, dict) or any(not _finite(losses.get(name)) for name in _LOSS_KEYS):
        return 'missing' if not isinstance(losses, dict) or any(name not in losses for name in _LOSS_KEYS) else 'nonfinite'
    origin = losses['origin']
    a_a = (losses['A_plus'] - losses['A_minus']) / 2
    a_b = (losses['B_plus'] - losses['B_minus']) / 2
    k_a = losses['A_plus'] + losses['A_minus'] - 2 * origin
    k_b = losses['B_plus'] + losses['B_minus'] - 2 * origin
    k_ab = losses['full_step'] - origin - a_a - a_b - k_a / 2 - k_b / 2
    values = (a_a, a_b, k_a, k_b, k_ab)
    if any(not _finite(value) for value in values):
        return 'nonfinite'
    return {'origin': origin, 'aA': a_a, 'aB': a_b, 'kA': k_a, 'kB': k_b, 'kAB': k_ab,
            'predicted_opposite': origin + a_a - a_b + k_a / 2 + k_b / 2 - k_ab,
            'actual_opposite': losses['opposite_mix']}


def _joint(coefficients):
    determinant = coefficients['kA'] * coefficients['kB'] - coefficients['kAB'] ** 2
    if not _finite(determinant) or determinant == 0:
        return None
    gradient_a, gradient_b = coefficients['aA'], coefficients['aB']
    step_a = -(coefficients['kB'] * gradient_a - coefficients['kAB'] * gradient_b) / determinant
    step_b = (coefficients['kAB'] * gradient_a - coefficients['kA'] * gradient_b) / determinant
    if not _finite(step_a) or not _finite(step_b):
        return None
    return step_a, step_b, determinant


def _exact(bank):
    slopes = bank.get('exact_slopes')
    if not isinstance(slopes, dict) or type(slopes.get('missing_gradient')) is not bool:
        return 'missing'
    if slopes.get('nonfinite') is True:
        return 'nonfinite'
    if slopes['missing_gradient']:
        return 'missing_gradient'
    if any(not _finite(slopes.get(name)) for name in ('A', 'B')):
        return 'nonfinite' if any(name in slopes for name in ('A', 'B')) else 'missing'
    return slopes


def _response(bank):
    output = bank.get('output') if isinstance(bank, dict) else None
    if not isinstance(output, dict):
        return 'missing', None
    needed = ('rms_dA', 'rms_dB', 'rms_dAB', 'rms_residual', 'ms_cross')
    if any(name not in output for name in needed):
        return 'missing', None
    if any(not _finite(output.get(name)) for name in needed):
        return 'nonfinite', None
    total = output['rms_dAB']
    forbidden = not total > 0 or output['rms_residual'] > 0.5 * total
    return None, {'forbidden': forbidden, **{name: output[name] for name in needed}}


def _adam(block):
    if not isinstance(block, dict):
        return 'missing'
    if block.get('status') in ('unsupported', 'nonfinite'):
        return 'fail'
    required = ('relative_rms', 'contaminated_slope_fraction', 'step', 'weight_decay', 'amsgrad', 'eps', 'lr')
    if any(name not in block for name in required):
        return 'missing'
    relative, fraction = block['relative_rms'], block['contaminated_slope_fraction']
    if any(not _finite(value) for value in (relative, fraction, block['eps'], block['lr'])):
        return 'fail'
    passed = (block['step'] == 1 and block['weight_decay'] == 0 and block['amsgrad'] is False
              and block['eps'] == 1e-8 and block['lr'] == 2e-4
              and relative <= 1e-4 and fraction <= 0.01)
    return None if passed else 'fail'


def _separable(coefficients, exact):
    return (-coefficients['aA'] / coefficients['kA'], -coefficients['aB'] / coefficients['kB'],
            -exact['A'] / coefficients['kA'], -exact['B'] / coefficients['kB'])


def _bank_identity(probe):
    failure = probe.get('failure') if isinstance(probe.get('failure'), dict) else {}
    if failure.get('stage') == 'bank_identity':
        return True
    hashes = probe.get('bank_hashes')
    if not isinstance(hashes, dict):
        return False
    values = [hashes.get(name) for name in ('monitor', 'fitting_0', 'fitting_1')]
    if any(not isinstance(value, str) or not value for value in values):
        return False
    return len(set(values)) < 3


def _abstain_nonfinite(probe, **detail):
    return _abstain(probe, 'nonfinite', **detail)


def _earliest_gate(coefficients, exact):
    """First numbered gate this bank fails, before cross-bank comparisons."""
    if not (coefficients['kA'] > 0 and coefficients['kB'] > 0):
        return 'curvature', None
    joint = _joint(coefficients)
    coefficients['joint'] = None if joint is None else {'sA': joint[0], 'sB': joint[1], 'determinant': joint[2]}
    if joint is None or not joint[2] > 0:
        return 'hessian', None
    if exact == 'missing_gradient' or not (coefficients['aA'] < 0 and coefficients['aB'] < 0
                                            and exact['A'] < 0 and exact['B'] < 0):
        return 'slopes', None
    factors = _separable(coefficients, exact)
    if any(not _finite(value) for value in factors):
        return 'nonfinite', None
    if any(not _unit_interval(value) for value in factors):
        return 'separable_factors', factors
    coefficients['symmetric'] = {'sA': factors[0], 'sB': factors[1]}
    coefficients['exact_factors'] = {'sA': factors[2], 'sB': factors[3]}
    return None, factors


def decide(probe):
    """Return one proposal or an abstention from a probe object."""
    if not isinstance(probe, dict):
        return _abstain({}, 'incomplete_evidence', reason='Probe evidence is not an object')
    if probe.get('contained_nonfinite') is True:
        return _abstain_nonfinite(probe)
    if _bank_identity(probe):
        return _abstain(probe, 'bank_identity')
    group_a = probe.get('group_a')
    group_b = probe.get('group_b')
    if group_a != list(GROUP_A) or not isinstance(group_b, list) or not group_b:
        return _abstain(probe, 'group_identity')
    if any(not isinstance(path, str) or not path for path in group_b) or len(set(group_b)) != len(group_b):
        return _abstain(probe, 'group_identity')
    if set(group_b) & set(GROUP_A):
        return _abstain(probe, 'group_identity')
    banks = probe.get('banks')
    if not isinstance(banks, list) or len(banks) != 2:
        return _abstain(probe, 'incomplete_evidence', reason='Two fitting banks are required')
    ranked = []
    for bank in banks:
        status, detail = _rank_bank(bank, set(GROUP_A))
        if status == 'nonfinite':
            return _abstain_nonfinite(probe)
        if status:
            return _abstain(probe, status, confirmation=detail)
        if detail['top4'] != list(GROUP_A) and set(detail['top4']) != set(GROUP_A):
            return _abstain(probe, 'group_not_confirmed', confirmation=detail)
        if set(detail['top4']) != set(GROUP_A) or detail['share'] < 0.70:
            return _abstain(probe, 'group_not_confirmed', confirmation=detail)
        ranked.append(detail)
    fits, exacts, responses = [], [], []
    for bank in banks:
        coefficients = _coefficients(bank.get('losses') if isinstance(bank, dict) else None)
        if coefficients == 'nonfinite':
            return _abstain_nonfinite(probe, confirmation=ranked)
        if coefficients == 'missing':
            return _abstain(probe, 'incomplete_evidence', reason='Stencil losses are missing', confirmation=ranked)
        exact = _exact(bank)
        if exact == 'nonfinite':
            return _abstain_nonfinite(probe, confirmation=ranked)
        if exact == 'missing':
            return _abstain(probe, 'incomplete_evidence', reason='Exact group slopes are missing', confirmation=ranked)
        status, response = _response(bank)
        if status == 'nonfinite':
            return _abstain_nonfinite(probe, confirmation=ranked)
        if status:
            return _abstain(probe, 'incomplete_evidence', reason='Group output responses are missing', confirmation=ranked)
        fits.append(coefficients)
        exacts.append(exact)
        responses.append(response)
    judged = [_earliest_gate(coefficients, exact) for coefficients, exact in zip(fits, exacts)]
    if any(gate == 'nonfinite' for gate, _factors in judged):
        return _abstain_nonfinite(probe, coefficients=fits, confirmation=ranked)
    for gate in ('curvature', 'hessian', 'slopes', 'separable_factors'):
        if any(item == gate for item, _factors in judged):
            return _abstain(probe, gate, coefficients=fits, confirmation=ranked)
    separable = [factors for _gate, factors in judged]
    for index in (0, 1):
        if abs(separable[0][index] - separable[1][index]) > 0.05:
            return _abstain(probe, 'banks', symmetric=[{'sA': item[0], 'sB': item[1]} for item in separable],
                            confirmation=ranked)
    for coefficients, factors in zip(fits, separable):
        joint = _joint(coefficients)
        if (joint is None or not _unit_interval(joint[0]) or not _unit_interval(joint[1])
                or abs(joint[0] - factors[0]) > 0.05 or abs(joint[1] - factors[1]) > 0.05):
            return _abstain(probe, 'cross_term', coefficients=fits, confirmation=ranked)
    for coefficients in fits:
        change = coefficients['actual_opposite'] - coefficients['origin']
        error = abs(coefficients['predicted_opposite'] - coefficients['actual_opposite'])
        if not (_finite(change) and change != 0 and error <= 0.5 * abs(change)):
            return _abstain(probe, 'unused_point', coefficients=fits, confirmation=ranked)
    adam = probe.get('adam') if isinstance(probe.get('adam'), dict) else {}
    adam_results = {name: _adam(adam.get(name)) for name in ('A', 'B')}
    if any(result == 'missing' for result in adam_results.values()):
        return _abstain(probe, 'incomplete_evidence', reason='Adam proportionality record is missing', confirmation=ranked)
    if any(result == 'fail' for result in adam_results.values()):
        return _abstain(probe, 'adam_step_not_proportional', adam_checks=adam_results, confirmation=ranked)
    expected_budget = {'native_updates': 1, 'loss_evaluations': 14,
                       'projection_backwards': 8, 'output_forwards': 4}
    recorded_cost = _cost(probe)
    if (any(recorded_cost[name] != value for name, value in expected_budget.items())
            or recorded_cost['elapsed_seconds'] is None):
        return _abstain(probe, 'incomplete_evidence', reason='Probe budget does not match the declared stencil',
                        confirmation=ranked)
    emitted = [min(separable[0][index], separable[1][index]) for index in (0, 1)]
    if emitted[0] == 1 and emitted[1] == 1:
        return _abstain(probe, 'no_change', symmetric=[{'sA': item[0], 'sB': item[1]} for item in separable],
                        confirmation=ranked, response=responses)
    rules = []
    if emitted[0] != 1:
        rules.extend({'pattern': path, 'multiplier': emitted[0]} for path in GROUP_A)
    if emitted[1] != 1:
        rules.extend({'pattern': path, 'multiplier': emitted[1]} for path in group_b)
    return {'schema_version': 1, 'layer_lr_multipliers': rules, 'evidence': {
        'decision': 'propose', 'formula': _FORMULA, 'assumptions': [
            'FLeRM supplies the response attribution, not a base profile to match.',
            'GeN supplies the signed stationary factor on one actual Adam direction. No periodic refit or smoothing is applied.',
            'Independent multipliers are emitted only when the joint stationary point stays within 0.05 of the separable factors.',
        ],
        'group_a_multiplier': emitted[0], 'group_b_multiplier': emitted[1],
        'confirmation': ranked, 'coefficients': fits, 'response': responses,
        'response_decomposition_forbidden': any(item['forbidden'] for item in responses),
        **_cost(probe),
    }}


def stencil_allowed(probe):
    """True only when both fitting banks confirm the declared four-tensor group."""
    if not isinstance(probe, dict) or probe.get('group_a') != list(GROUP_A):
        return False
    banks = probe.get('banks')
    if not isinstance(banks, list) or len(banks) != 2:
        return False
    for bank in banks:
        status, detail = _rank_bank(bank, set(GROUP_A))
        if status or not isinstance(detail, dict) or detail.get('share') is None:
            return False
        if set(detail['top4']) != set(GROUP_A) or detail['share'] < 0.70:
            return False
    return True


def propose(context):
    evidence = context.get('evidence') if isinstance(context, dict) else None
    if not isinstance(evidence, list) or len(evidence) != 1:
        return _abstain({}, 'incomplete_evidence', reason='Exactly one probe report is required')
    return decide(evidence[0])
