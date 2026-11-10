"""`checkpoints prune` safety: run lock, run-directory checks, symlinks and failed deletes."""
import json
import os
from pathlib import Path

import pytest
import torch

from hypergan import checkpoints
from hypergan.checkpoints import _prune_run_checkpoints, extract_ema_bundle, prune_checkpoints
from hypergan.run_state import run_lock

POLICY = {'keep_last': 1, 'keep_every': 0, 'keep_best': None}


def make_checkpoint(run_dir, attempt, step, suffix=None, sequence=None):
    path = Path(run_dir) / 'checkpoints' / f'{attempt}-step-{step:08d}-{suffix or format(step, "012x")}'
    path.mkdir(parents=True)
    torch.save({'step': step}, path / 'state.pt')
    info = {'run_id': 'run', 'attempt_id': attempt, 'step': step}
    if sequence is not None:
        info['event_boundary'] = {'sequence': sequence}
    (path / 'manifest.json').write_text(json.dumps(info))
    return path


def point_latest(run_dir, path, step):
    (Path(run_dir) / 'checkpoints' / 'latest.json').write_text(json.dumps(
        {'schema_version': 1, 'checkpoint': path.name, 'step': step}))


def write_attempt(run_dir, attempt, parent=None, checkpoint_path=None, current=False):
    manifest = {'run_id': 'run', 'attempt_id': attempt, 'status': 'complete', 'config': {'checkpoints': POLICY}}
    if parent is not None:
        manifest['recovery_parent'] = parent
    if checkpoint_path is not None:
        manifest['checkpoint_path'] = str(checkpoint_path)
    paths = [Path(run_dir) / 'attempts' / attempt / 'manifest.json']
    if current:
        paths.append(Path(run_dir) / 'manifest.json')
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest))


def simple_run(run_dir, steps=(0, 1, 2, 3)):
    for step in steps:
        latest = make_checkpoint(run_dir, '0001-a', step)
    point_latest(run_dir, latest, steps[-1])
    write_attempt(run_dir, '0001-a', checkpoint_path=latest, current=True)
    return latest


def names(run_dir):
    return sorted(path.name for path in (Path(run_dir) / 'checkpoints').iterdir() if path.name != 'latest.json')


@pytest.mark.parametrize('evidence', ['sequence', 'manifest'])
def test_abandoned_attempt_keeps_its_last_written_checkpoint_when_two_share_its_last_step(tmp_path, evidence):
    # Heavy-suite failure (test_interval_resume_and_older_snapshot_recovery_keep_attempt_positions):
    # a resumed attempt wrote a periodic and a final checkpoint at step 5. After
    # a recovery from step 2, prune kept the abandoned attempt's step-5
    # checkpoint with the larger random suffix and deleted the final one the
    # resume had returned.
    by_sequence = evidence == 'sequence'
    make_checkpoint(tmp_path, '0001-a', 0)
    origin = make_checkpoint(tmp_path, '0001-a', 2)
    write_attempt(tmp_path, '0001-a', checkpoint_path=origin)
    parent = {'attempt_id': '0001-a', 'step': 2, 'checkpoint_id': origin.name}
    for step in (3, 4):
        make_checkpoint(tmp_path, '0002-b', step, sequence=step if by_sequence else None)
    periodic = make_checkpoint(tmp_path, '0002-b', 5, suffix='f' * 12, sequence=5 if by_sequence else None)
    final = make_checkpoint(tmp_path, '0002-b', 5, suffix='0' * 12, sequence=6 if by_sequence else None)
    write_attempt(tmp_path, '0002-b', parent=parent, checkpoint_path=None if by_sequence else final)
    for step in (3, 4, 5):
        latest = make_checkpoint(tmp_path, '0003-c', step)
    point_latest(tmp_path, latest, 5)
    write_attempt(tmp_path, '0003-c', parent=parent, checkpoint_path=latest, current=True)

    report = prune_checkpoints(tmp_path)
    reasons = {item['name']: item['reasons'] for item in report['retained']}
    assert 'abandoned_tip' in reasons[final.name]
    assert final.is_dir() and not periodic.exists()


def test_prune_refuses_a_directory_that_is_not_a_run_or_a_directory_of_runs(tmp_path):
    target = tmp_path / 'notes'
    (target / 'drafts').mkdir(parents=True)
    (target / 'drafts' / 'keep.txt').write_text('mine')
    with pytest.raises(ValueError, match='Not a HyperGAN run directory'):
        prune_checkpoints(target)
    assert (target / 'drafts' / 'keep.txt').read_text() == 'mine'
    # A missing path has nothing to delete and still reports an empty prune.
    assert prune_checkpoints(tmp_path / 'missing')['deleted_count'] == 0


def test_prune_refuses_a_run_whose_lock_is_held(tmp_path):
    simple_run(tmp_path)
    before = names(tmp_path)
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match='already locked'):
            prune_checkpoints(tmp_path)
        assert names(tmp_path) == before
        # The trainer prunes under the lock it already holds.
        assert _prune_run_checkpoints(tmp_path, policy=POLICY)['deleted_count'] == 2
    assert len(names(tmp_path)) == 2


def test_prune_of_a_runs_root_skips_active_runs_and_symlinked_children(tmp_path):
    root = tmp_path / 'runs'
    simple_run(root / 'idle')
    simple_run(root / 'active')
    outside = tmp_path / 'elsewhere'
    simple_run(outside)
    os.symlink(outside, root / 'linked', target_is_directory=True)
    active_before = names(root / 'active')
    with run_lock(root / 'active'):
        report = prune_checkpoints(root, policy=POLICY)
    assert report['skipped'] == [{'run_dir': str((root / 'active').resolve()), 'reason': 'locked'}]
    assert report['deleted_count'] == 2 and len(names(root / 'idle')) == 2
    assert names(root / 'active') == active_before
    assert len(names(outside)) == 4


def test_prune_never_deletes_through_symlinks(tmp_path):
    outside = tmp_path / 'elsewhere'
    simple_run(outside)
    run = tmp_path / 'run'
    run.mkdir()
    os.symlink(outside / 'checkpoints', run / 'checkpoints', target_is_directory=True)
    (run / 'manifest.json').write_text(json.dumps({'run_id': 'run', 'attempt_id': '0001-a'}))
    with pytest.raises(ValueError, match='symlinked checkpoints directory'):
        prune_checkpoints(run, policy=POLICY)
    assert len(names(outside)) == 4

    # A single symlinked checkpoint entry is kept and its target untouched.
    other = tmp_path / 'other'
    target = make_checkpoint(other, '0009-z', 1)
    real = tmp_path / 'real'
    simple_run(real, steps=(0, 2, 3))
    os.symlink(target, real / 'checkpoints' / '0001-a-step-00000001-linked000000', target_is_directory=True)
    report = prune_checkpoints(real, policy=POLICY)
    reasons = {item['name']: item['reasons'] for item in report['retained']}
    assert 'symlink' in reasons['0001-a-step-00000001-linked000000']
    assert (target / 'state.pt').is_file() and (target / 'manifest.json').is_file()


def test_failed_checkpoint_delete_is_reported_and_not_counted_as_reclaimed(tmp_path, monkeypatch):
    simple_run(tmp_path)
    stuck = next(path for path in (tmp_path / 'checkpoints').iterdir() if '-step-00000001-' in path.name)
    real_rmtree = checkpoints.shutil.rmtree

    def rmtree(path, ignore_errors=False, **kwargs):
        # Behaves like an undeletable directory: silent with ignore_errors, raising without.
        if Path(path) == stuck:
            if ignore_errors:
                return
            raise PermissionError(13, 'Permission denied', str(path))
        return real_rmtree(path, ignore_errors=ignore_errors, **kwargs)

    monkeypatch.setattr(checkpoints.shutil, 'rmtree', rmtree)
    report = prune_checkpoints(tmp_path, policy=POLICY)
    assert [item['step'] for item in report['deleted']] == [2]
    assert [item['name'] for item in report['failed']] == [stuck.name]
    assert 'PermissionError' in report['failed'][0]['error']
    assert report['failed_count'] == 1 and report['deleted_count'] == 1
    assert report['reclaimed_bytes'] == report['deleted'][0]['size']


def test_extract_ema_bundle_requires_the_checkpoints_own_manifest(tmp_path):
    # The removed fallback read checkpoints/manifest.json, a file a run never writes.
    ckpt = tmp_path / 'checkpoints' / '0001-a-step-00000001-000000000001'
    ckpt.mkdir(parents=True)
    torch.save({'step': 1}, ckpt / 'state.pt')
    (tmp_path / 'checkpoints' / 'manifest.json').write_text(json.dumps({'config': {'components': {}}}))
    with pytest.raises(ValueError, match='Checkpoint manifest missing'):
        extract_ema_bundle(ckpt)
