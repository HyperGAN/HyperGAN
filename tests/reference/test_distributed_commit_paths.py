"""Actual Linux special-file checks for the parent commit byte reader."""
import os
from pathlib import Path

import pytest

import hypergan.distributed_commit as commit


@pytest.mark.parametrize('kind', ['symlink', 'fifo', 'directory'])
def test_nonregular_payload_is_rejected_before_open(tmp_path, monkeypatch, kind):
    path = tmp_path / 'rank.pt'
    if kind == 'symlink':
        target = tmp_path / 'elsewhere'
        target.write_text('payload')
        path.symlink_to(target)
    elif kind == 'fifo':
        os.mkfifo(path)
    else:
        path.mkdir()
    def forbidden(*args, **kwargs):
        raise AssertionError('special-file rejection attempted a potentially blocking open')
    monkeypatch.setattr(commit.os, 'open', forbidden)
    with pytest.raises(ValueError, match='ordinary files'):
        commit._read_regular(path)


def test_managed_staging_ancestor_symlink_is_rejected(tmp_path):
    root = tmp_path / 'distributed-checkpoints'
    root.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (root / '.prepared').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='ordinary'):
        commit.preparation_directory(tmp_path, 'attempt', 'controller', 1, 'a' * 12, create=True)
    assert not list(outside.iterdir())
