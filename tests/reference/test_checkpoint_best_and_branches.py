"""One shared "best" ranking, branch-aware retention and checkpoint write hygiene."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hypergan import checkpoints
from hypergan.checkpoints import prune_checkpoints, resolve_checkpoint, write_checkpoint
from hypergan.metrics import digest
from hypergan.summary import read_summary_json, write_summary_json


def make_checkpoint(run_dir, attempt, step):
    path = Path(run_dir) / 'checkpoints' / f'{attempt}-step-{step:08d}-{step:012x}'
    path.mkdir(parents=True)
    torch.save({'step': step}, path / 'state.pt')
    (path / 'manifest.json').write_text(json.dumps({'run_id': 'run', 'attempt_id': attempt, 'step': step}))
    return path


def point_latest(run_dir, path):
    (Path(run_dir) / 'checkpoints' / 'latest.json').write_text(json.dumps(
        {'schema_version': 1, 'checkpoint': path.name, 'step': int(path.name.split('-step-')[1][:8])}))


def write_catalog(run_dir, definitions):
    metrics = {}
    for name, fields in definitions.items():
        definition = {'kind': 'scalar', 'source': 'custom:' + name, **fields}
        metrics[name] = dict(definition, definition_hash=digest(definition))
    catalog = {'schema_version': 1, 'metrics': metrics}
    revision = digest(catalog)
    path = Path(run_dir) / 'metrics' / f'catalog-{revision}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog))
    return revision


def write_receipt(run_dir, metric, step, value, catalog, attempt='0001-a'):
    directory = Path(run_dir) / 'metrics' / 'evaluations' / f'step-{step:08d}-{metric}-{step:08x}'
    directory.mkdir(parents=True)
    (directory / 'receipt.json').write_text(json.dumps({
        'schema_version': 1, 'status': 'complete', 'metric_id': metric, 'source_step': step,
        'attempt_id': attempt, 'catalog': catalog, 'result': {'value': value, 'step': step}}))


def write_manifest(run_dir, attempt, policy, parent=None, attempt_dir=None):
    manifest = {'run_id': 'run', 'attempt_id': attempt, 'status': 'complete', 'steps': 0,
                'config': {'checkpoints': policy}}
    if parent is not None:
        manifest['recovery_parent'] = parent
    for path in [Path(run_dir) / 'manifest.json'] + ([Path(run_dir) / 'attempts' / attempt_dir / 'manifest.json'] if attempt_dir else []):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest))


def steps(report, key):
    return sorted((item['name'].split('-step-')[0], item['step']) for item in report[key])


def test_best_checkpoint_keep_best_and_summary_agree_on_a_maximize_metric(tmp_path):
    # Review 2026-10-25 #9: with only a "higher is better" metric, keep_best used
    # to pool it under the default min mode and pin the two worst checkpoints.
    for step in (0, 100, 200, 300, 400, 500):
        latest = make_checkpoint(tmp_path, '0001-a', step)
    point_latest(tmp_path, latest)
    write_manifest(tmp_path, '0001-a', {'keep_last': 1, 'keep_every': 0}, attempt_dir='0001-a')
    revision = write_catalog(tmp_path, {'modes_covered': {'direction': 'maximize', 'unit': 'modes'}})
    for step, value in [(100, 5.0), (200, 9.0), (300, 7.0), (400, 2.0)]:
        write_receipt(tmp_path, 'modes_covered', step, value, revision)
    report = prune_checkpoints(tmp_path, dry_run=True)
    assert steps(report, 'deleted') == [('0001-a', 100), ('0001-a', 400)]
    best = {item['step'] for item in report['retained'] if 'best(modes_covered)' in item['reasons']}
    assert best == {200, 300}
    assert resolve_checkpoint(tmp_path, 'best').name.startswith('0001-a-step-00000200-')
    write_summary_json(tmp_path)
    summary = read_summary_json(tmp_path)
    assert (summary['best']['step'], summary['best']['direction']) == (200, 'maximize')


def test_best_uses_only_the_configured_fid_protocol(tmp_path):
    # A 128-sample smoke FID is not interchangeable with fid50k_train.
    for step in (0, 10, 20, 30):
        latest = make_checkpoint(tmp_path, '0001-a', step)
    point_latest(tmp_path, latest)
    write_manifest(tmp_path, '0001-a', {'keep_last': 1, 'keep_every': 0,
                   'keep_best': {'metric': 'fid50k_train', 'mode': 'min', 'count': 1}}, attempt_dir='0001-a')
    revision = write_catalog(tmp_path, {name: {'direction': 'minimize', 'unit': 'FID'}
                                        for name in ('fid_smoke', 'fid50k_train')})
    write_receipt(tmp_path, 'fid_smoke', 10, 3.0, revision)
    write_receipt(tmp_path, 'fid50k_train', 10, 40.0, revision)
    write_receipt(tmp_path, 'fid50k_train', 20, 30.0, revision)
    assert resolve_checkpoint(tmp_path, 'best').name.startswith('0001-a-step-00000020-')
    report = prune_checkpoints(tmp_path, dry_run=True)
    assert steps(report, 'deleted') == [('0001-a', 10)]


def test_best_skips_evaluations_without_a_checkpoint_and_matches_the_attempt(tmp_path):
    for attempt, step in [('0001-a', 0), ('0001-a', 10), ('0002-b', 10), ('0002-b', 20)]:
        latest = make_checkpoint(tmp_path, attempt, step)
    point_latest(tmp_path, latest)
    write_manifest(tmp_path, '0002-b', {}, parent={'attempt_id': '0001-a', 'step': 0, 'checkpoint_id': 'x'})
    revision = write_catalog(tmp_path, {'fid50k_train': {'direction': 'minimize', 'unit': 'FID'}})
    write_receipt(tmp_path, 'fid50k_train', 15, 1.0, revision, attempt='0002-b')   # no checkpoint at 15
    write_receipt(tmp_path, 'fid50k_train', 10, 2.0, revision, attempt='0001-a')
    write_receipt(tmp_path, 'fid50k_train', 20, 3.0, revision, attempt='0002-b')
    assert resolve_checkpoint(tmp_path, 'best').name.startswith('0001-a-step-00000010-')


def branch_fixture(run_dir, *, record_first_attempt=True):
    # Mirrors review 2026-10-25 slice 2 M2: keep_last=2, keep_every=4, train to
    # 12, then resume from step 4 for three updates.
    policy = {'keep_last': 2, 'keep_every': 4, 'keep_best': None}
    for step in (0, 4, 8, 11, 12):
        make_checkpoint(run_dir, '0001-a', step)
    if record_first_attempt:
        write_manifest(run_dir, '0001-a', policy, attempt_dir='0001-a')
    for step in (4, 5, 6):
        latest = make_checkpoint(run_dir, '0002-b', step)
    point_latest(run_dir, latest)
    parent = {'attempt_id': '0001-a', 'step': 4, 'checkpoint_id': '0001-a-step-00000004-000000000004'}
    write_manifest(run_dir, '0002-b', policy, parent=parent, attempt_dir='0002-b')


def test_retention_keeps_the_live_branch_after_resuming_from_an_older_checkpoint(tmp_path):
    branch_fixture(tmp_path)
    report = prune_checkpoints(tmp_path)
    # The live branch keeps its last two; the abandoned 11 and 12 no longer
    # count as latest or last. The abandoned tip (12, also a milestone) and
    # the milestone 8 survive; 11 is pruned.
    assert steps(report, 'deleted') == [('0001-a', 11)]
    assert steps(report, 'retained') == [('0001-a', 0), ('0001-a', 4), ('0001-a', 8), ('0001-a', 12),
                                         ('0002-b', 4), ('0002-b', 5), ('0002-b', 6)]
    assert sorted(path.name.split('-step-')[1][:8] for path in (tmp_path / 'checkpoints').glob('0002-b-*')) == [
        '00000004', '00000005', '00000006']


def test_unknown_lineage_falls_back_to_keeping_every_branch(tmp_path):
    branch_fixture(tmp_path, record_first_attempt=False)
    report = prune_checkpoints(tmp_path, dry_run=True)
    assert ('0001-a', 12) in steps(report, 'retained') and ('0001-a', 11) in steps(report, 'retained')


@pytest.fixture
def small_state(monkeypatch):
    monkeypatch.setattr(checkpoints, 'trainer_state', lambda trainer, last_batch: {'step': trainer.step, 'w': torch.arange(1000.)})


def test_checkpoint_write_hashes_while_writing_and_sweeps_stale_pending(tmp_path, small_state, monkeypatch):
    root = tmp_path / 'checkpoints'
    stale = root / '.pending-0001-a-step-00000003-0123456789ab'
    stale.mkdir(parents=True)
    (stale / 'state.pt').write_bytes(b'x' * 4096)
    unrelated = root / '.pending-notes'
    unrelated.mkdir()
    def reread(path):
        raise AssertionError('state.pt must not be re-read to hash it')
    monkeypatch.setattr(checkpoints, 'file_sha256', reread)
    target = write_checkpoint(tmp_path, SimpleNamespace(step=4), None, {'attempt_id': '0001-a'})
    info = json.loads((target / 'manifest.json').read_text())
    assert info['state_sha256'] == hashlib.sha256((target / 'state.pt').read_bytes()).hexdigest()
    assert torch.load(target / 'state.pt', weights_only=True)['step'] == 4
    assert not stale.exists()
    assert unrelated.is_dir()
    assert sorted(path.name for path in root.iterdir()) == sorted(['.pending-notes', 'latest.json', target.name])


@pytest.mark.heavy
def test_real_resume_from_older_checkpoint_keeps_live_branch(tmp_path):
    from tests.config_fixtures import write_default
    from hypergan.training import resume, train
    path = write_default(tmp_path / 'config.toml', device='cpu', defaults='hypergan')
    text = path.read_text().replace('steps = 5', 'steps = 12').replace('num_particles = 20000', 'num_particles = 32')
    path.write_text(text.replace('count = 256', 'count = 4'))
    run = tmp_path / 'run'
    first = train(path, run, checkpoint_every=1)['attempt_id']
    resumed = resume(run, checkpoint=resolve_checkpoint(run, 10), stop_after_steps=1)['attempt_id']
    kept = {(p.name.split('-step-')[0], int(p.name.split('-step-')[1][:8]))
            for p in (run / 'checkpoints').iterdir() if p.is_dir()}
    # Default keep_last=3 counts the live branch 0 -> 10 -> 11. The abandoned
    # steps 11 and 12 no longer count as "latest" or "last": only the abandoned
    # tip 12 stays, so the old branch remains resumable.
    assert kept == {(first, 0), (first, 10), (first, 12), (resumed, 10), (resumed, 11)}
