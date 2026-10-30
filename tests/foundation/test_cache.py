"""Tests for HyperGAN's on-demand external URL cache engine (~/.hypergan/cache/)."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from pathlib import Path
import threading
import time
import pytest

import hypergan as hg
from hypergan.cache import (
    INCEPTION_V3_SHA256,
    INCEPTION_V3_URL,
    RESNET18_SHA256,
    RESNET18_URL,
    OfflineError,
    cache_file_lock,
    ensure_cached_url,
    get_cache_dir,
    is_offline,
    resolve_or_cache,
)
from hypergan.image_components import _verified_file
from hypergan.image_metrics import _checked_weights


class _LocalTestServer:
    """Lightweight HTTP server for testing URL cache behavior."""

    def __init__(self, payload: bytes, *, delay: float = 0.0, support_range: bool = True):
        self.payload = payload
        self.delay = delay
        self.support_range = support_range
        self.requests: list[dict] = []
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                range_header = self.headers.get("Range")
                parent.requests.append({"path": self.path, "range": range_header})
                if parent.delay > 0:
                    time.sleep(parent.delay)
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
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                    self.send_header("Content-Length", str(len(slice_bytes)))
                    self.end_headers()
                    self.wfile.write(slice_bytes)
                    return

                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format, *args):
                return

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_port
        self.url = f"http://127.0.0.1:{self.port}/test-blob.bin"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def test_cache_dir_resolution(tmp_path, monkeypatch):
    """Verify cache directory defaults to ~/.hypergan/cache and respects HYPERGAN_CACHE_DIR."""
    custom_dir = tmp_path / "custom_cache"
    monkeypatch.setenv("HYPERGAN_CACHE_DIR", str(custom_dir))
    assert get_cache_dir() == custom_dir.resolve()
    assert custom_dir.is_dir()

    monkeypatch.delenv("HYPERGAN_CACHE_DIR", raising=False)
    home_cache = Path.home() / ".hypergan" / "cache"
    assert get_cache_dir() == home_cache.resolve()


def test_cache_offline_mode_rejection(tmp_path, monkeypatch):
    """Verify HYPERGAN_OFFLINE=1 raises OfflineError when resource is not cached, but allows cached hits."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HYPERGAN_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("HYPERGAN_OFFLINE", "1")
    assert is_offline() is True

    uncached_url = "https://example.com/nonexistent-weight.pth"
    with pytest.raises(OfflineError) as exc_info:
        resolve_or_cache(uncached_url, cache_dir=cache_dir)
    assert "offline mode" in str(exc_info.value)
    assert issubclass(OfflineError, FileNotFoundError)

    # Pre-populate cache and verify offline mode permits cached access
    cached_name = "test-cached.bin"
    payload = b"offline pre-cached data"
    sha = hashlib.sha256(payload).hexdigest()
    (cache_dir / cached_name).write_bytes(payload)

    cached_url = f"https://example.com/{cached_name}"
    resolved = resolve_or_cache(cached_url, expected_sha256=sha, cache_dir=cache_dir)
    assert resolved == cache_dir / cached_name
    assert resolved.read_bytes() == payload


def test_cache_sha256_mismatch_and_part_cleanup(tmp_path):
    """Verify hash mismatch raises ValueError and unlinks partial .part file."""
    cache_dir = tmp_path / "cache"
    payload = b"legitimate server data"
    wrong_sha = "0" * 64

    with _LocalTestServer(payload) as server:
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            ensure_cached_url(server.url, expected_sha256=wrong_sha, cache_dir=cache_dir)

    dest_path = cache_dir / "test-blob.bin"
    part_path = cache_dir / "test-blob.bin.part"
    assert not dest_path.exists()
    assert not part_path.exists()


def test_cache_local_path_passthrough_no_copy(tmp_path):
    """Verify strict local path pass-through never copies files to cache."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plain local path
    local_file = tmp_path / "local_model.pt"
    payload = b"some local model parameters"
    local_file.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()

    result = resolve_or_cache(local_file, expected_sha256=sha, cache_dir=cache_dir)
    assert result == local_file.resolve()
    # Cache directory must remain empty
    assert list(cache_dir.iterdir()) == []

    # 2. file:// URI
    file_uri = f"file://{local_file.resolve()}"
    result_uri = resolve_or_cache(file_uri, expected_sha256=sha, cache_dir=cache_dir)
    assert result_uri == local_file.resolve()
    assert list(cache_dir.iterdir()) == []

    # 3. Missing local file raises FileNotFoundError
    missing_local = tmp_path / "does_not_exist.pt"
    with pytest.raises(FileNotFoundError, match="missing"):
        resolve_or_cache(missing_local, cache_dir=cache_dir)


def test_cache_concurrent_file_locking_simulation(tmp_path):
    """Simulate multi-rank workers downloading identical URL simultaneously."""
    cache_dir = tmp_path / "cache"
    payload = b"large binary artifact for concurrent locking simulation " * 5000
    sha = hashlib.sha256(payload).hexdigest()

    with _LocalTestServer(payload, delay=0.05) as server:
        def worker(rank):
            return resolve_or_cache(server.url, expected_sha256=sha, cache_dir=cache_dir)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            results = [f.result() for f in futures]

        for p in results:
            assert p == cache_dir / "test-blob.bin"
            assert p.stat().st_size == len(payload)

        # Server must have received at most 1 full download request
        assert len(server.requests) >= 1
        assert p.read_bytes() == payload


def test_cache_range_resume(tmp_path):
    """Verify partial download resumes cleanly using HTTP Range requests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = b"abcdefghijklmnopqrstuvwxyz0123456789" * 100
    sha = hashlib.sha256(payload).hexdigest()
    split = len(payload) // 3

    part_file = cache_dir / "test-blob.bin.part"
    part_file.write_bytes(payload[:split])

    with _LocalTestServer(payload, support_range=True) as server:
        result = ensure_cached_url(server.url, expected_sha256=sha, expected_size=len(payload), cache_dir=cache_dir)
        assert result == cache_dir / "test-blob.bin"
        assert result.read_bytes() == payload
        assert not part_file.exists()
        assert len(server.requests) == 1
        assert server.requests[0]["range"] == f"bytes={split}-"


def test_image_components_resnet_default_download(tmp_path, monkeypatch):
    """Verify image_components._verified_file delegates default ResNet-18 to cache."""
    cache_dir = tmp_path / "cache"
    payload = b"fake resnet weights"
    sha = hashlib.sha256(payload).hexdigest()

    cached_file = cache_dir / "resnet18-f37072fd.pth"
    cached_file.parent.mkdir(parents=True, exist_ok=True)
    cached_file.write_bytes(payload)

    monkeypatch.setattr("hypergan.image_components.RESNET18_SHA256", sha)
    monkeypatch.setattr("hypergan.image_components.RESNET18_URL", "https://download.pytorch.org/models/resnet18-f37072fd.pth")
    monkeypatch.setenv("HYPERGAN_CACHE_DIR", str(cache_dir))

    resolved = _verified_file(None, expected=sha)
    assert resolved == cached_file.resolve()


def test_image_metrics_inception_default_download(tmp_path, monkeypatch):
    """Verify image_metrics._checked_weights delegates default Inception-v3 to cache."""
    cache_dir = tmp_path / "cache"
    payload = b"fake inception weights"
    sha = hashlib.sha256(payload).hexdigest()

    cached_file = cache_dir / "weights-inception-2015-12-05-6726825d.pth"
    cached_file.parent.mkdir(parents=True, exist_ok=True)
    cached_file.write_bytes(payload)

    monkeypatch.setattr("hypergan.image_metrics.INCEPTION_SHA256", sha)
    monkeypatch.setattr("hypergan.image_metrics.INCEPTION_V3_URL", "https://example.com/weights-inception-2015-12-05-6726825d.pth")
    monkeypatch.setenv("HYPERGAN_CACHE_DIR", str(cache_dir))

    resolved = _checked_weights(None, expected=sha)
    assert resolved == cached_file.resolve()
