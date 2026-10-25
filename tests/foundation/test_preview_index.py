"""The published preview index is portable: its paths are relative to the run."""
import json
from pathlib import Path, PureWindowsPath
import shutil

from hypergan import previews
from hypergan.previews import drain_pruning, index_path, publish_preview_payload

ATTEMPT = '0001-' + 'a' * 32


def publish(run_dir, sequence, step):
    identity = {'run_id': 'run-one', 'attempt_id': ATTEMPT, 'attempt_index': 1,
                'sample_sequence': sequence, 'name': 'g'}
    payload = {'schema_version': 1, 'kind': 'ema-preview', 'identity': identity, 'name': 'g',
               'step': step, 'count': 1, 'shape': [1, 1], 'samples': [[0.5]]}
    return publish_preview_payload(run_dir, payload, identity, step)


def test_index_paths_are_relative_and_resolve_after_the_run_moves(tmp_path):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    publish(run_dir, 1, 1)
    record, index, _ = publish(run_dir, 2, 2)
    # The caller still receives where the files are now.
    assert Path(record['path']).is_absolute() and Path(record['path']).is_file()
    assert [Path(item['path']).is_file() for item in index['previews']] == [True, True]

    published = json.loads((run_dir / 'previews' / 'index.json').read_text())
    paths = [item['path'] for item in published['previews']]
    assert all(not Path(path).is_absolute() and path.startswith('previews/') for path in paths)
    assert paths[-1] == Path(record['path']).relative_to(run_dir).as_posix()

    moved = tmp_path / 'elsewhere' / 'run'
    moved.parent.mkdir()
    shutil.move(run_dir, moved)
    for path in paths:
        assert index_path(moved, path).is_file()
    # Publishing in the new place reuses the relative index and keeps history.
    _, index, _ = publish(moved, 3, 3)
    assert [item['identity']['sample_sequence'] for item in index['previews']] == [1, 2, 3]
    assert all(Path(item['path']).is_file() for item in index['previews'])
    drain_pruning(5)


def test_older_absolute_index_entries_are_mapped_onto_the_run_directory(tmp_path):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    record, _, _ = publish(run_dir, 1, 1)
    generation = Path(record['path']).parent.name
    # An index written before paths were relative, for a run since copied.
    legacy = Path(tmp_path.anchor, 'somewhere', 'old', 'run', 'previews', generation, 'preview.json')
    assert index_path(run_dir, str(legacy)) == run_dir / 'previews' / generation / 'preview.json'
    assert index_path(run_dir, str(legacy)).is_file()
    # A path outside any preview generation is left as recorded.
    outside = Path(tmp_path.anchor, 'data', 'other', 'file.json')
    assert index_path(run_dir, str(outside)) == outside


def test_rooted_paths_without_a_drive_never_resolve_against_the_run(monkeypatch):
    # On Windows `/x` has a root but no drive, so `is_absolute()` is False.
    monkeypatch.setattr(previews, 'Path', PureWindowsPath)
    generation = '000000000001-0001-' + 'a' * 32 + '-step00000001-' + 'b' * 32
    run_dir = PureWindowsPath('C:/runs/run')
    legacy = f'/somewhere/old/run/previews/{generation}/preview.json'
    assert index_path(run_dir, legacy) == run_dir / 'previews' / generation / 'preview.json'
    assert index_path(run_dir, '/data/other/file.json') == PureWindowsPath('/data/other/file.json')
    assert index_path(run_dir, 'previews/x/preview.json') == run_dir / 'previews' / 'x' / 'preview.json'
