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
