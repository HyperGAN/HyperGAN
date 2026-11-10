"""DINOv3 source checkouts: cached Git verification and per-checkout imports.

Each checkout is a small local Git repository with the upstream package layout;
nothing is downloaded.
"""
import subprocess
import sys
import types

import pytest
from hndl import Registry

from hypergan import pretrained_providers

BACKBONES = '''
import torch


def dinov3_vits16(*, pretrained=True):
    assert pretrained is False
    # Upstream imports model modules lazily, at construction.
    from ..models import marker
    model = torch.nn.Identity()
    model.marker = marker.MARKER
    return model
'''


def _git(source, *args):
    return subprocess.check_output(
        ['git', '-C', str(source), '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid',
         '-c', 'commit.gpgsign=false', *args], text=True).strip()


def _commit(source, message='fixture'):
    _git(source, 'add', '-A')
    _git(source, 'commit', '-q', '-m', message)
    # Refresh the index once, so later status calls find nothing to rewrite.
    _git(source, 'status', '--porcelain')
    return _git(source, 'rev-parse', 'HEAD')


def _checkout(path, marker):
    files = {'.gitignore': '__pycache__/\n', 'dinov3/__init__.py': '', 'dinov3/hub/__init__.py': '',
             'dinov3/hub/backbones.py': BACKBONES, 'dinov3/models/__init__.py': '',
             'dinov3/models/marker.py': f'MARKER = {marker!r}\n'}
    for name, text in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text)
    subprocess.check_output(['git', 'init', '-q', str(path)])
    return str(path), _commit(path)


def _register(source, commit):
    registry = pretrained_providers.register_providers(
        Registry.builtins(), {'dinov3_vits16': {'source_path': source, 'source_commit': commit}})
    return registry.pretrained_builder('dinov3_vits16')


@pytest.fixture(autouse=True)
def fresh_caches(monkeypatch):
    monkeypatch.setattr(pretrained_providers, '_VERIFIED', {})
    monkeypatch.setattr(pretrained_providers, '_PACKAGES', {})


@pytest.fixture
def git_calls(monkeypatch):
    calls = []
    original = subprocess.check_output

    def counted(command, *args, **kwargs):
        calls.append(command)
        return original(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'check_output', counted)
    return calls


def test_unchanged_checkout_is_verified_once(tmp_path, git_calls):
    source, commit = _checkout(tmp_path / 'a', 'a')
    git_calls.clear()
    for _ in range(3):
        assert _register(source, commit)().marker == 'a'
    assert [command[3:5] for command in git_calls] == [['rev-parse', 'HEAD'], ['status', '--porcelain']]
    assert not any(pretrained_providers._is_dinov3(name) for name in sys.modules)


def test_staged_change_and_new_commit_are_verified_again(tmp_path):
    root = tmp_path / 'a'
    source, commit = _checkout(root, 'a')
    _register(source, commit)
    (root / 'dinov3' / 'extra.py').write_text('')
    _git(root, 'add', '-A')
    with pytest.raises(ValueError, match='dirty=True'):
        _register(source, commit)
    (root / 'dinov3' / 'models' / 'marker.py').write_text("MARKER = 'a2'\n")
    second = _commit(root, 'second')
    with pytest.raises(ValueError, match=f'got {second}'):
        _register(source, commit)
    # The new revision imports its own modules, not the ones cached for the first.
    assert _register(source, second)().marker == 'a2'


def test_two_checkouts_each_build_from_their_own_package(tmp_path):
    first = _register(*_checkout(tmp_path / 'a', 'a'))
    second = _register(*_checkout(tmp_path / 'b', 'b'))
    assert [build().marker for build in (first, second, first)] == ['a', 'b', 'a']
    assert not any(pretrained_providers._is_dinov3(name) for name in sys.modules)


def test_an_already_imported_dinov3_is_set_aside_and_restored(tmp_path, monkeypatch):
    installed = types.ModuleType('dinov3')
    installed.__file__ = str(tmp_path / 'site-packages' / 'dinov3' / '__init__.py')
    monkeypatch.setitem(sys.modules, 'dinov3', installed)
    assert _register(*_checkout(tmp_path / 'a', 'a'))().marker == 'a'
    assert sys.modules['dinov3'] is installed
