#!/usr/bin/env python3
"""Run one declared startup algorithm through a shared, disposable evaluator.

PYTHONPATH=src python research/startup_tuning/benchmark.py MANIFEST --case source \
    --device cuda:0 --output-root /path/benchmark

A manifest lists explicit cases, never a search space. Algorithms return JSON
proposals; the evaluator owns training and all recorded outcome measurements.
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'reports'))


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path):
    path = Path(path).expanduser().resolve()
    manifest = read_json(path)
    allowed = {'schema_version', 'name', 'config', 'evaluation', 'cases', 'baseline'}
    if set(manifest) - allowed or manifest.get('schema_version') != 1:
        raise ValueError('Unsupported benchmark manifest schema or fields')
    if not isinstance(manifest.get('cases'), list) or not manifest['cases']:
        raise ValueError('Declare at least one explicit case')
    cases = []
    for reference in manifest['cases']:
        if not isinstance(reference, str):
            raise ValueError('Each case must reference a separate tuning solution JSON file')
        solution_path = (path.parent / reference).resolve()
        case = read_json(solution_path)
        if case.pop('schema_version', None) != 1:
            raise ValueError('Unsupported tuning solution schema')
        if set(case) - {'id', 'algorithm', 'options', 'evidence', 'description'}:
            raise ValueError('Unsupported solution fields (seed overrides are not supported)')
        case['_solution_path'] = str(solution_path)
        case['_solution_sha256'] = sha256(solution_path)
        cases.append(case)
    manifest['cases'] = cases
    ids = [case.get('id') for case in cases]
    if any(not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', name) for name in ids) or len(set(ids)) != len(ids):
        raise ValueError('Case IDs must be unique safe directory names')
    if manifest.get('baseline') not in ids:
        raise ValueError('Declare a baseline case ID')
    evaluation = manifest.get('evaluation', {})
    if set(evaluation) - {'steps', 'observation_steps', 'features', 'direction', 'crossed'}:
        raise ValueError('Unsupported evaluation fields')
    return path, manifest


def proposal_for(case, config, manifest_path):
    """One algorithm invocation on serialized inputs, without evaluator access.

    Custom Python is trusted local research code. It receives a copied config,
    options and explicitly supplied probe evidence, never the monitor bank or
    a live training object. There is no retry, seed override, or search loop.
    """
    evidence, provenance = [], []
    solution_path = Path(case['_solution_path'])
    for reference in case.get('evidence', []):
        path = (solution_path.parent / reference).resolve()
        evidence.append(read_json(path))
        provenance.append({'path': str(path), 'sha256': sha256(path)})
    name = case.get('algorithm', 'source')
    options = copy.deepcopy(case.get('options', {}))
    source = {'name': name, 'evidence': provenance, 'solution_path': str(solution_path),
              'solution_sha256': case['_solution_sha256']}
    started = time.monotonic()
    if name == 'source':
        if options:
            raise ValueError('The source baseline accepts no tuning options')
        plan = {'schema_version': 1}
    elif name == 'fixed':
        plan = {'schema_version': 1, **options}
    else:
        path = (solution_path.parent / name).resolve()
        if path.suffix != '.py':
            raise ValueError('Custom algorithm must name a local .py file exposing propose(context)')
        source['sha256'] = sha256(path)
        source['path'] = str(path)
        spec = importlib.util.spec_from_file_location('startup_benchmark_algorithm', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        plan = module.propose({'config': copy.deepcopy(config), 'options': options,
                               'evidence': evidence})
    json.dumps(plan, allow_nan=False)
    if not isinstance(plan, dict):
        raise ValueError('Algorithm must return a JSON proposal object')
    return plan, {**source, 'proposal_seconds': time.monotonic() - started}


def run_case(manifest_path, manifest, case, output_root, device):
    from hypergan.config import load_config
    from joint_rate_probe import run_probe
    from proposals import apply_proposal

    destination = output_root / case['id']
    destination.mkdir(parents=True, exist_ok=False)
    config_path = (manifest_path.parent / Path(manifest['config']).expanduser()).resolve()
    config = load_config(config_path)
    (destination / 'training-config.toml').write_bytes(config_path.read_bytes())
    (destination / 'resolved-training-config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
    request = {'manifest': manifest, 'manifest_path': str(manifest_path),
               'manifest_sha256': sha256(manifest_path), 'case': case['id'],
               'algorithm': {'name': case.get('algorithm', 'source'),
                             'solution_path': case['_solution_path'],
                             'solution_sha256': case['_solution_sha256']}}
    (destination / 'request.json').write_text(json.dumps(request, indent=2, allow_nan=False) + '\n')
    (destination / 'solution.json').write_text(Path(case['_solution_path']).read_text())
    began = time.monotonic()
    try:
        plan, algorithm = proposal_for(case, config, manifest_path)
    except Exception as failure:
        # A failed formula is still an experiment outcome. No trainer has been
        # constructed at this point; do not fabricate restoration/quality data.
        report = {'schema_version': 2, 'kind': 'disposable-explicit-joint-rate-rollout',
                  'status': 'failed', 'completed_updates': 0,
                  'requested_updates': manifest.get('evaluation', {}).get('steps', 32),
                  'observations': [], 'per_step': [], 'budget': {'completed_native_training_updates': 0},
                  'proposal': {'case': case['id'], 'algorithm': request['algorithm']},
                  'failure': {'stage': 'proposal', 'type': type(failure).__name__, 'message': str(failure)},
                  'elapsed_seconds': time.monotonic() - began,
                  'interpretation': ['Proposal creation failed before constructing a trainer; no quality or state audit measurements.']}
        (destination / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        raise
    request.update(algorithm=algorithm, proposal=plan)
    (destination / 'request.json').write_text(json.dumps(request, indent=2, allow_nan=False) + '\n')
    if algorithm.get('path'):
        (destination / 'algorithm.py').write_bytes(Path(algorithm['path']).read_bytes())

    @contextmanager
    def prepare(trainer):
        with apply_proposal(trainer, plan) as resolved:
            yield {'case': case['id'], 'algorithm': algorithm, 'resolved': resolved,
                   'manifest_sha256': request['manifest_sha256']}

    evaluation = manifest.get('evaluation', {})
    report = run_probe(config_path, g_lr=plan.get('g_lr', config['optimizer']['lr']),
                       d_lr=plan.get('d_lr', config['optimizer']['lr'] * config['optimizer']['d_lr_mult']),
                       steps=evaluation.get('steps', 32), device=device,
                       features=evaluation.get('features', False),
                       direction=evaluation.get('direction', False),
                       crossed=evaluation.get('crossed', False),
                       observe_steps=evaluation.get('observation_steps'),
                       prepare=prepare, progress_path=destination / 'report.json')
    if report.get('failure'):
        raise RuntimeError(f"Case {case['id']} failed; see {destination / 'report.json'}")
    return destination / 'report.json'


def update_leaderboard(manifest, output_root):
    from leaderboard import build_leaderboard, markdown
    baseline = output_root / manifest['baseline'] / 'report.json'
    if not baseline.exists() or read_json(baseline).get('status') != 'complete':
        return
    paths = [output_root / case['id'] / 'report.json' for case in manifest['cases']
             if case['id'] != manifest['baseline']]
    paths = [path for path in paths if path.exists() and read_json(path).get('status') in ('complete', 'failed')]
    board = build_leaderboard(baseline, paths)
    (output_root / 'leaderboard.json').write_text(json.dumps(board, indent=2, allow_nan=False) + '\n')
    (output_root / 'leaderboard.md').write_text(markdown(board))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--case', required=True, help='One case ID, or all to run the finite manifest sequentially')
    parser.add_argument('--device')
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    path, manifest = load_manifest(args.manifest)
    cases = [case for case in manifest['cases'] if args.case == 'all' or case['id'] == args.case]
    if not cases:
        parser.error('Unknown case ID')
    output_root = Path(args.output_root).expanduser().resolve()
    for case in cases:
        print(f"benchmark {manifest['name']} | {case['id']}", flush=True)
        try:
            print(run_case(path, manifest, case, output_root, args.device), flush=True)
        finally:
            update_leaderboard(manifest, output_root)


if __name__ == '__main__':
    main()
