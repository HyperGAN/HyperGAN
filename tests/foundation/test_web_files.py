import hashlib
import os

import pytest

from hypergan.web_files import read_artifact, read_bytes, read_json


def test_indexed_content_and_bounds(tmp_path):
    (tmp_path / "samples").mkdir()
    data = b'{"modality":"tensor","shape":[2,2]}\n'
    (tmp_path / "samples" / "example.json").write_bytes(data)
    record = dict(path="samples/example.json", bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    assert read_artifact(tmp_path, record) == data
    assert read_json(tmp_path, record['path'])['modality'] == 'tensor'
    with pytest.raises(ValueError, match="byte budget"):
        read_bytes(tmp_path, record['path'], max_bytes=1)
    (tmp_path / record['path']).write_bytes(b'bad')
    with pytest.raises(ValueError, match="changed"):
        read_artifact(tmp_path, record)


@pytest.mark.parametrize('path', ['../outside', '/etc/passwd', 'samples/../file',
                                 'samples//file', './file', 'C:/file', 'samples\\file'])
def test_path_input_rejected(tmp_path, path):
    with pytest.raises(ValueError, match="relative path"):
        read_bytes(tmp_path, path)


def test_parent_and_final_symlinks_do_not_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("private")
    root = tmp_path / "run"
    root.mkdir()
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        # Windows CI does not always grant symlink creation. Exercise an invalid
        # native path there; POSIX performs both real link escape checks below.
        if os.name != "nt":
            raise
        with pytest.raises(ValueError):
            read_bytes(root, "../outside/secret")
        return
    (root / "file").symlink_to(outside / "secret")
    for path in ["link/secret", "file"]:
        with pytest.raises(ValueError):
            read_bytes(root, path)


@pytest.mark.parametrize('data', [b'{"a": NaN}', b'{"a": 1e999}'])
def test_nonfinite_json_rejected(tmp_path, data):
    (tmp_path / "bad.json").write_bytes(data)
    with pytest.raises(ValueError):
        read_json(tmp_path, 'bad.json')
