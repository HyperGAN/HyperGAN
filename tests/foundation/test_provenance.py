"""Release identities remain useful outside a Git checkout."""

import json
from pathlib import Path
import subprocess

import pytest

from hypergan.provenance import hypergan_source


def _git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True,
                                   stderr=subprocess.DEVNULL).strip()


def test_source_and_editable_identity_observe_clean_dirty_and_untracked_files(tmp_path):
    package = tmp_path / 'src' / 'hypergan'
    package.mkdir(parents=True)
    source = package / '__init__.py'
    source.write_text('VERSION = 1\n')
    _git(tmp_path, 'init')
    _git(tmp_path, 'add', '.')
    _git(tmp_path, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
         'commit', '-m', 'fixture')
    commit = _git(tmp_path, 'rev-parse', 'HEAD')
    assert hypergan_source(package) == {'hypergan_commit': commit, 'hypergan_dirty': False,
                                       'hypergan_provenance': 'git'}
    source.write_text('VERSION = 2\n')
    assert hypergan_source(package)['hypergan_dirty'] is True
    _git(tmp_path, 'checkout', '--', 'src/hypergan/__init__.py')
    (package / 'untracked.py').write_text('')
    assert hypergan_source(package)['hypergan_dirty'] is True
    assert hypergan_source(package)['hypergan_commit'] == commit


@pytest.mark.parametrize('dirty', [True, False, None])
def test_wheel_and_unpacked_sdist_use_stamped_identity_without_git(tmp_path, dirty):
    package = tmp_path / 'src' / 'hypergan'
    package.mkdir(parents=True)
    (package / '_build_provenance.json').write_text(json.dumps({
        'schema_version': 1, 'commit': 'a' * 40, 'dirty': dirty,
    }))
    assert hypergan_source(package) == {'hypergan_commit': 'a' * 40,
                                       'hypergan_dirty': dirty, 'hypergan_provenance': 'build'}


@pytest.mark.parametrize('stamp', [None, '{', '[]', '{"schema_version": true}',
    '{"schema_version": 1, "commit": "not-a-commit", "dirty": false}',
    '{"schema_version": 1, "commit": null, "dirty": null}',
    '{"schema_version": 1, "commit": "' + 'a' * 40 + '", "dirty": 1}'])
def test_missing_or_invalid_provenance_is_explicitly_unknown(tmp_path, stamp):
    package = tmp_path / 'src' / 'hypergan'
    package.mkdir(parents=True)
    if stamp is not None:
        (package / '_build_provenance.json').write_text(stamp)
    assert hypergan_source(package) == {'hypergan_commit': None, 'hypergan_dirty': None,
                                       'hypergan_provenance': 'unknown'}


def test_current_distribution_has_a_release_revision():
    source = hypergan_source()
    assert source['hypergan_commit'] is not None
    assert source['hypergan_provenance'] in {'git', 'build'}
    assert type(source['hypergan_dirty']) is bool


def test_write_launch_provenance_schema_and_atomicity(tmp_path):
    from hypergan.provenance import write_launch_provenance

    run_dir = tmp_path / "run_01"
    meta = {
        "run": {"env": "develop", "devices": ["gpu0"]},
        "devices": [{"index": 0, "name": "RTX 4090", "alias": "gpu0"}],
        "git": {"particlegan": {"commit": "1234567890abcdef1234567890abcdef12345678", "dirty": False}},
    }
    target = write_launch_provenance(run_dir, attempt=1, metadata=meta)
    assert target.is_file()
    assert target.name == "0001.json"
    assert target.parent == run_dir / "launch"

    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["attempt"] == 1
    assert isinstance(data["argv"], list)
    assert isinstance(data["timestamp"], str)
    assert data["git"]["particlegan"]["commit"] == "1234567890abcdef1234567890abcdef12345678"
    assert data["run"]["env"] == "develop"
    assert len(data["devices"]) == 1

    # Incremental attempt
    target2 = write_launch_provenance(run_dir, attempt=2, metadata=meta)
    assert target2.name == "0002.json"
    assert (run_dir / "launch" / "0001.json").is_file()


def test_snapshot_config_sources_preserves_content(tmp_path):
    from hypergan.provenance import snapshot_config_sources

    run_dir = tmp_path / "run_01"
    src1 = tmp_path / "cifar-base.toml"
    src1.write_text("# Base CIFAR comments\nformat = 'hypergan-model/1'\n")
    src2 = tmp_path / "child.toml"
    src2.write_text("# Child experiment comments\nextends = 'cifar-base.toml'\n")

    snapshot_dir = snapshot_config_sources(run_dir, [src1, src2])
    assert snapshot_dir == run_dir / "config-source"
    assert (snapshot_dir / "cifar-base.toml").read_text() == src1.read_text()
    assert (snapshot_dir / "child.toml").read_text() == src2.read_text()


def test_inspect_git_repository_live_and_non_git(tmp_path):
    from hypergan.provenance import inspect_git_repository

    # Non-git directory
    res = inspect_git_repository(tmp_path)
    assert res == {'commit': None, 'dirty': None, 'provenance': 'unknown'}

    # Git directory
    _git(tmp_path, 'init')
    (tmp_path / 'foo.txt').write_text('bar\n')
    _git(tmp_path, 'add', '.')
    _git(tmp_path, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
         'commit', '-m', 'fixture')
    res2 = inspect_git_repository(tmp_path)
    assert res2['provenance'] == 'git'
    assert len(res2['commit']) == 40
    assert res2['dirty'] is False

