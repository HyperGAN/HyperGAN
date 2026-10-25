"""Tests for opt-in dataset acquisition, checksum verification, resume, and extraction."""

from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import tarfile
import threading

import pytest

import hypergan as hg
from hypergan import dataset_download
from hypergan.cli import main as cli_main


def _build_synthetic_cifar_archive(
    archive_path: Path,
    *,
    corrupt_batch: str | None = None,
    extra_member: tarfile.TarInfo | None = None,
    extra_payload: bytes = b"",
) -> tuple[int, str, dict[str, str]]:
    """Create a deterministic synthetic CIFAR-10 tar.gz archive and return (size, sha256, files_sha256)."""
    files_sha256: dict[str, str] = {}
    with tarfile.open(archive_path, mode="w:gz") as tar:
        dir_info = tarfile.TarInfo(name="cifar-10-batches-py")
        dir_info.type = tarfile.DIRTYPE
        dir_info.mode = 0o755
        tar.addfile(dir_info)

        names = [*(f"data_batch_{i}" for i in range(1, 6)), "test_batch"]
        for idx, name in enumerate(names):
            payload = f"synthetic-cifar10-{name}-payload-{idx * 101}\n".encode("utf-8") * 64
            digest = hashlib.sha256(payload).hexdigest()
            files_sha256[name] = digest
            written_payload = b"corrupted-batch-bytes" if corrupt_batch == name else payload
            info = tarfile.TarInfo(name=f"cifar-10-batches-py/{name}")
            info.size = len(written_payload)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(written_payload))

        if extra_member is not None:
            tar.addfile(extra_member, io.BytesIO(extra_payload) if extra_member.isfile() else None)

    raw = archive_path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest(), files_sha256


class _RangeArchiveServer:
    """Local HTTP server supporting Range (206 Partial Content) and request counting."""

    def __init__(self, payload: bytes, *, support_range: bool = True, truncate_at: int | None = None):
        self.payload = payload
        self.support_range = support_range
        self.truncate_at = truncate_at
        self.requests: list[dict] = []
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                range_header = self.headers.get("Range")
                parent.requests.append({"path": self.path, "range": range_header})
                data = parent.payload
                if range_header and parent.support_range and range_header.startswith("bytes="):
                    start_str = range_header[len("bytes="):].split("-", 1)[0]
                    start = int(start_str)
                    if start >= len(data):
                        self.send_response(416)
                        self.end_headers()
                        return
                    slice_bytes = data[start:]
                    self.send_response(206)
                    self.send_header("Content-Type", "application/gzip")
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                    self.send_header("Content-Length", str(len(slice_bytes)))
                    self.end_headers()
                    if parent.truncate_at is not None:
                        slice_bytes = slice_bytes[:parent.truncate_at]
                    self.wfile.write(slice_bytes)
                    return

                self.send_response(200)
                self.send_header("Content-Type", "application/gzip")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                body = data[:parent.truncate_at] if parent.truncate_at is not None else data
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_port
        self.url = f"http://127.0.0.1:{self.port}/cifar-10-python.tar.gz"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def test_list_and_info_metadata_without_network(capsys):
    datasets = hg.list_datasets()
    assert len(datasets) == 1
    info = hg.dataset_info("cifar10")
    assert info["name"] == "cifar10"
    assert info["version"] == "python-1.0"
    assert info["url"] == "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    assert info["expected_bytes"] == 170_498_071
    assert info["archive_sha256"] == "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
    assert set(info["files_sha256"]) == {*(f"data_batch_{i}" for i in range(1, 6)), "test_batch"}
    assert "Krizhevsky" in info["citation"]
    assert "research" in info["usage_notice"]

    rc = cli_main(["download-data", "--list", "--json"])
    assert rc == 0
    listed_json = json.loads(capsys.readouterr().out)
    assert [d["name"] for d in listed_json["datasets"]] == ["cifar10"]

    rc = cli_main(["download-data", "cifar10", "--info"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" in out
    assert "170498071 bytes" in out
    assert "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce" in out


@pytest.mark.parametrize("unqualified", ["mnist", "celeba", "imagenet", "coco"])
def test_unqualified_datasets_rejected_clearly(unqualified, capsys):
    with pytest.raises(ValueError, match="qualified datasets: cifar10") as exc_info:
        hg.dataset_info(unqualified)
    assert "image_folder" in str(exc_info.value)

    rc = cli_main(["download-data", unqualified, "--info"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "qualified datasets: cifar10" in err
    assert "image_folder" in err


def test_download_extract_manifest_and_idempotency(tmp_path, monkeypatch, capsys):
    src_tar = tmp_path / "server_cifar.tar.gz"
    size, sha, files_sha = _build_synthetic_cifar_archive(src_tar)
    payload = src_tar.read_bytes()

    with _RangeArchiveServer(payload) as server:
        patched = deepcopy(dataset_download.QUALIFIED_DATASETS["cifar10"])
        patched.update(url=server.url, expected_bytes=size, archive_sha256=sha, files_sha256=files_sha)
        monkeypatch.setitem(dataset_download.QUALIFIED_DATASETS, "cifar10", patched)

        dest = tmp_path / "data"
        rc = cli_main(["download-data", "cifar10", "--output", str(dest), "--json"])
        assert rc == 0
        result = json.loads(capsys.readouterr().out)

        assert result["dataset"] == "cifar10"
        assert result["verified"] is True
        assert result["reused_extracted"] is False
        assert result["reused_archive"] is False
        assert result["archive_sha256"] == sha
        assert result["archive_bytes"] == size
        assert len(server.requests) == 1

        manifest_path = dest / "dataset-manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 1
        assert manifest["dataset"] == "cifar10"
        assert Path(manifest["root"]) == (dest / "cifar-10-batches-py").resolve()
        assert manifest["data_config"]["factory"] == "hypergan.datasets:CIFAR10Data"
        assert "hypergan.datasets:CIFAR10Data" in manifest["config_snippet"]
        assert len(manifest["files"]) == 6

        # Second call reuses the verified extracted directory without making network requests.
        second = hg.download_dataset("cifar10", dest)
        assert second["reused_extracted"] is True
        assert len(server.requests) == 1

        # Verify-only mode succeeds without network requests.
        verified = hg.download_dataset("cifar10", dest, verify_only=True)
        assert verified["verified"] is True
        assert len(server.requests) == 1

        # New destination sharing the same cache directory reuses the cached archive without network requests.
        dest2 = tmp_path / "data2"
        third = hg.download_dataset("cifar10", dest2, cache_dir=dest / ".cache")
        assert third["reused_extracted"] is False
        assert third["reused_archive"] is True
        assert len(server.requests) == 1
        assert (dest2 / "cifar-10-batches-py" / "data_batch_1").is_file()


def test_interrupted_part_file_resumes_via_http_range(tmp_path, monkeypatch):
    src_tar = tmp_path / "server_cifar.tar.gz"
    size, sha, files_sha = _build_synthetic_cifar_archive(src_tar)
    payload = src_tar.read_bytes()

    with _RangeArchiveServer(payload) as server:
        patched = deepcopy(dataset_download.QUALIFIED_DATASETS["cifar10"])
        patched.update(url=server.url, expected_bytes=size, archive_sha256=sha, files_sha256=files_sha)
        monkeypatch.setitem(dataset_download.QUALIFIED_DATASETS, "cifar10", patched)

        dest = tmp_path / "resumed_data"
        cache_dir = dest / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        part_file = cache_dir / "cifar-10-python.tar.gz.part"
        split_offset = size // 3
        part_file.write_bytes(payload[:split_offset])

        result = hg.download_dataset("cifar10", dest)
        assert result["resumed_partial"] is True
        assert result["verified"] is True
        assert not part_file.exists()
        assert (cache_dir / "cifar-10-python.tar.gz").is_file()
        assert len(server.requests) == 1
        assert server.requests[0]["range"] == f"bytes={split_offset}-"


def test_corrupted_checksum_and_truncated_download_rejected(tmp_path, monkeypatch):
    src_tar = tmp_path / "server_cifar.tar.gz"
    size, sha, files_sha = _build_synthetic_cifar_archive(src_tar)
    payload = src_tar.read_bytes()

    # 1. Mismatched archive SHA256 must fail and clean up .part file.
    with _RangeArchiveServer(payload) as server:
        patched = deepcopy(dataset_download.QUALIFIED_DATASETS["cifar10"])
        patched.update(
            url=server.url,
            expected_bytes=size,
            archive_sha256="0" * 64,
            files_sha256=files_sha,
        )
        monkeypatch.setitem(dataset_download.QUALIFIED_DATASETS, "cifar10", patched)

        dest = tmp_path / "bad_sha_dest"
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            hg.download_dataset("cifar10", dest)
        assert not (dest / ".cache" / "cifar-10-python.tar.gz.part").exists()
        assert not (dest / "cifar-10-batches-py").exists()

    # 2. Truncated HTTP response must fail.
    with _RangeArchiveServer(payload, truncate_at=size // 2) as server:
        patched = deepcopy(dataset_download.QUALIFIED_DATASETS["cifar10"])
        patched.update(url=server.url, expected_bytes=size, archive_sha256=sha, files_sha256=files_sha)
        monkeypatch.setitem(dataset_download.QUALIFIED_DATASETS, "cifar10", patched)

        dest_trunc = tmp_path / "trunc_dest"
        with pytest.raises(OSError, match="Truncated download"):
            hg.download_dataset("cifar10", dest_trunc)
        assert not (dest_trunc / "cifar-10-batches-py").exists()


def test_malicious_tar_members_and_corrupt_batch_rejected(tmp_path, monkeypatch):
    # 1. Path traversal '../evil' member inside archive.
    traversal_tar = tmp_path / "traversal.tar.gz"
    evil = tarfile.TarInfo(name="cifar-10-batches-py/../../evil.txt")
    evil.size = 4
    evil.mode = 0o644
    size, sha, files_sha = _build_synthetic_cifar_archive(
        traversal_tar, extra_member=evil, extra_payload=b"evil"
    )
    patched = deepcopy(dataset_download.QUALIFIED_DATASETS["cifar10"])
    patched.update(expected_bytes=size, archive_sha256=sha, files_sha256=files_sha)
    monkeypatch.setitem(dataset_download.QUALIFIED_DATASETS, "cifar10", patched)

    dest = tmp_path / "traversal_dest"
    with pytest.raises(ValueError, match="Unsafe path traversal"):
        hg.download_dataset("cifar10", dest, source_archive=traversal_tar)
    assert not (tmp_path / "evil.txt").exists()
    assert not (dest / "cifar-10-batches-py").exists()

    # 2. Symlink member inside archive.
    symlink_tar = tmp_path / "symlink.tar.gz"
    sym = tarfile.TarInfo(name="cifar-10-batches-py/link")
    sym.type = tarfile.SYMTYPE
    sym.linkname = "/etc/passwd"
    size, sha, files_sha = _build_synthetic_cifar_archive(symlink_tar, extra_member=sym)
    patched.update(expected_bytes=size, archive_sha256=sha, files_sha256=files_sha)
    monkeypatch.setitem(dataset_download.QUALIFIED_DATASETS, "cifar10", patched)

    dest_sym = tmp_path / "sym_dest"
    with pytest.raises(ValueError, match="Unsafe tar member type"):
        hg.download_dataset("cifar10", dest_sym, source_archive=symlink_tar)
    assert not (dest_sym / "cifar-10-batches-py").exists()

    # 3. Corrupted inner batch file rejected before publishing extracted directory.
    corrupt_batch_tar = tmp_path / "corrupt_batch.tar.gz"
    size, sha, files_sha = _build_synthetic_cifar_archive(
        corrupt_batch_tar, corrupt_batch="data_batch_3"
    )
    patched.update(expected_bytes=size, archive_sha256=sha, files_sha256=files_sha)
    monkeypatch.setitem(dataset_download.QUALIFIED_DATASETS, "cifar10", patched)

    dest_corrupt = tmp_path / "corrupt_dest"
    with pytest.raises(ValueError, match="Dataset file SHA256 mismatch for data_batch_3"):
        hg.download_dataset("cifar10", dest_corrupt, source_archive=corrupt_batch_tar)
    assert not (dest_corrupt / "cifar-10-batches-py").exists()
    assert not (dest_corrupt / "dataset-manifest.json").exists()


def test_symlinked_destination_and_source_archive_rejected(tmp_path, monkeypatch):
    archive = tmp_path / "real.tar.gz"
    size, sha, files_sha = _build_synthetic_cifar_archive(archive)
    patched = deepcopy(dataset_download.QUALIFIED_DATASETS["cifar10"])
    patched.update(expected_bytes=size, archive_sha256=sha, files_sha256=files_sha)
    monkeypatch.setitem(dataset_download.QUALIFIED_DATASETS, "cifar10", patched)
    real_dest = tmp_path / "real_dest"
    real_dest.mkdir()
    linked_dest = tmp_path / "linked_dest"
    linked_dest.symlink_to(real_dest, target_is_directory=True)
    with pytest.raises(ValueError, match="not a file or symlink"):
        hg.download_dataset("cifar10", linked_dest, source_archive=archive)
    assert list(real_dest.iterdir()) == []
    linked_archive = tmp_path / "linked.tar.gz"
    linked_archive.symlink_to(archive)
    with pytest.raises(FileNotFoundError, match="is a symlink"):
        hg.download_dataset("cifar10", tmp_path / "dest", source_archive=linked_archive)
    result = hg.download_dataset("cifar10", tmp_path / "dest", source_archive=archive)
    assert result["verified"] is True


def test_dataset_download_is_torch_free():
    import subprocess
    import sys

    src_dir = str(Path(__file__).resolve().parents[2] / "src")
    code = (
        "import sys; "
        f"sys.path.insert(0, {src_dir!r}); "
        "import hypergan as hg, hypergan.dataset_download; "
        "info = hg.dataset_info('cifar10'); "
        "assert info['name'] == 'cifar10'; "
        "assert 'torch' not in sys.modules; "
        "assert 'numpy' not in sys.modules; "
        "assert 'particlegan' not in sys.modules"
    )
    res = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=20)
    assert res.returncode == 0, res.stderr

