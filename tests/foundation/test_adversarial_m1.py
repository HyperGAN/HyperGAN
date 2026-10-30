"""Adversarial stress test suite for Milestone 1 (HyperGAN 2.0.0 GA).

Empirically verifies:
1. Cache concurrency (multi-process and multi-thread races on identical URLs).
2. Cache offline enforcement (HYPERGAN_OFFLINE=1 rejects downloads, passes existing cached files).
3. Cache hash corruption and partial HTTP Range resume / cleanup.
4. Cache local path rules (strict pass-through without copying or modifying cache dir).
5. Queue SIGTERM forwarding to child processes and bounded kill fallback.
6. Checkpoint pruning concurrent race stress with flock verification and invariant preservation.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import pytest

from hypergan.cache import (
    OfflineError,
    ensure_cached_url,
    get_cache_dir,
    is_offline,
    resolve_or_cache,
)
from hypergan.checkpoints import (
    _durable_checkpoints,
    _prune_lock,
    prune_checkpoints,
)
from hypergan.queue import (
    queue_add,
    queue_run,
    queue_status,
)


# ---------------------------------------------------------------------------
# Multiprocessing Top-Level Worker Callbacks (Must be picklable for spawn)
# ---------------------------------------------------------------------------

def _mp_cache_download_worker(url: str, expected_sha: str | None, cache_dir: str, result_queue):
    """Worker executed in a spawned child process for cache concurrency stress."""
    try:
        p = resolve_or_cache(url, expected_sha256=expected_sha, cache_dir=cache_dir)
        result_queue.put({
            "ok": True,
            "path": str(p),
            "size": p.stat().st_size,
            "pid": os.getpid(),
        })
    except Exception as exc:
        result_queue.put({
            "ok": False,
            "error": str(exc),
            "type": type(exc).__name__,
            "pid": os.getpid(),
        })


def _mp_prune_worker(run_dir_str: str, policy_dict: dict, result_queue):
    """Worker executed in a spawned child process for checkpoint pruning stress."""
    try:
        report = prune_checkpoints(run_dir_str, policy=policy_dict)
        result_queue.put({
            "ok": True,
            "retained": [r["step"] for r in report["retained"]],
            "deleted": [d["step"] for d in report["deleted"]],
            "reclaimed_bytes": report["reclaimed_bytes"],
            "pid": os.getpid(),
        })
    except Exception as exc:
        result_queue.put({
            "ok": False,
            "error": str(exc),
            "type": type(exc).__name__,
            "pid": os.getpid(),
        })


# ---------------------------------------------------------------------------
# Adversarial Mock HTTP Server
# ---------------------------------------------------------------------------

class _AdversarialHTTPServer:
    """Configurable HTTP server supporting range requests, artificial delays, and corruption."""

    def __init__(
        self,
        payloads: dict[str, bytes],
        *,
        delay: float = 0.0,
        support_range: bool = True,
        corrupt_range: bool = False,
    ):
        self.payloads = payloads
        self.delay = delay
        self.support_range = support_range
        self.corrupt_range = corrupt_range
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                with parent._lock:
                    range_header = self.headers.get("Range")
                    parent.requests.append({"path": self.path, "range": range_header})

                if parent.delay > 0:
                    time.sleep(parent.delay)

                path_clean = self.path.split("?")[0]
                if path_clean not in parent.payloads:
                    self.send_response(404)
                    self.end_headers()
                    return

                data = parent.payloads[path_clean]

                if range_header and parent.support_range and range_header.startswith("bytes="):
                    start_str = range_header[len("bytes="):].split("-", 1)[0]
                    start = int(start_str)
                    if start >= len(data):
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(data)}")
                        self.end_headers()
                        return

                    slice_bytes = data[start:]
                    if parent.corrupt_range:
                        slice_bytes = b"CORRUPTED_BYTES" + slice_bytes[15:]

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
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def url_for(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"http://127.0.0.1:{self.port}{path}"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


# ===========================================================================
# 1. Cache Concurrency Stress
# ===========================================================================

def test_cache_multiprocess_race_identical_url(tmp_path):
    """Adversarial stress: 8 independent OS processes simultaneously race to download the same URL."""
    cache_dir = tmp_path / "cache_mp"
    cache_dir.mkdir(parents=True, exist_ok=True)

    payload = b"ADVERSARIAL_CONCURRENCY_BLOB_PADDING_" * 15000  # ~555 KB
    expected_sha = hashlib.sha256(payload).hexdigest()

    with _AdversarialHTTPServer({"/model.pth": payload}, delay=0.03) as server:
        url = server.url_for("/model.pth")
        ctx = mp.get_context("spawn")
        q = ctx.Queue()

        num_procs = 8
        procs = [
            ctx.Process(target=_mp_cache_download_worker, args=(url, expected_sha, str(cache_dir), q))
            for _ in range(num_procs)
        ]

        for p in procs:
            p.start()

        for p in procs:
            p.join(timeout=15)
            assert not p.is_alive(), f"Process {p.pid} hung during concurrent cache download"

        results = [q.get(timeout=2) for _ in range(num_procs)]

        # Verify all processes succeeded without exception
        for res in results:
            assert res["ok"] is True, f"Subprocess download failed: {res}"
            assert res["size"] == len(payload)

        # Verify target file integrity
        final_file = cache_dir / "model.pth"
        assert final_file.is_file()
        assert final_file.stat().st_size == len(payload)
        assert hashlib.sha256(final_file.read_bytes()).hexdigest() == expected_sha

        # Verify no .part or lingering locks
        part_files = list(cache_dir.glob("*.part"))
        assert len(part_files) == 0, f"Lingering partial files found: {part_files}"

        # Under flock serialization, the server receives at most 2 requests (1 full, 0-1 race)
        assert len(server.requests) <= 3


def test_cache_multithread_race_identical_url(tmp_path):
    """Adversarial stress: 16 concurrent threads race on ensure_cached_url."""
    cache_dir = tmp_path / "cache_mt"
    cache_dir.mkdir(parents=True, exist_ok=True)

    payload = b"MULTITHREAD_STRESS_PAYLOAD_CONTENT_" * 10000
    expected_sha = hashlib.sha256(payload).hexdigest()

    with _AdversarialHTTPServer({"/threaded.bin": payload}, delay=0.02) as server:
        url = server.url_for("/threaded.bin")

        def worker():
            return ensure_cached_url(url, expected_sha256=expected_sha, cache_dir=cache_dir)

        num_threads = 16
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(worker) for _ in range(num_threads)]
            results = [f.result() for f in futures]

        for p in results:
            assert p == cache_dir / "threaded.bin"
            assert p.stat().st_size == len(payload)

        assert (cache_dir / "threaded.bin").read_bytes() == payload
        assert not (cache_dir / "threaded.bin.part").exists()


def test_cache_multiprocess_race_different_urls_same_dir(tmp_path):
    """Multiple processes downloading distinct URLs into the same cache directory concurrently."""
    cache_dir = tmp_path / "cache_diff"
    cache_dir.mkdir(parents=True, exist_ok=True)

    payload_a = b"PAYLOAD_A_DATA_" * 5000
    payload_b = b"PAYLOAD_B_DATA_" * 5000
    sha_a = hashlib.sha256(payload_a).hexdigest()
    sha_b = hashlib.sha256(payload_b).hexdigest()

    with _AdversarialHTTPServer({"/file_a.bin": payload_a, "/file_b.bin": payload_b}, delay=0.02) as server:
        url_a = server.url_for("/file_a.bin")
        url_b = server.url_for("/file_b.bin")

        ctx = mp.get_context("spawn")
        q = ctx.Queue()

        procs = []
        for i in range(6):
            target_url = url_a if i % 2 == 0 else url_b
            target_sha = sha_a if i % 2 == 0 else sha_b
            procs.append(ctx.Process(target=_mp_cache_download_worker, args=(target_url, target_sha, str(cache_dir), q)))

        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=10)

        results = [q.get(timeout=2) for _ in range(6)]
        for res in results:
            assert res["ok"] is True, f"Process failed: {res}"

        assert (cache_dir / "file_a.bin").read_bytes() == payload_a
        assert (cache_dir / "file_b.bin").read_bytes() == payload_b


# ===========================================================================
# 2. Cache Offline Enforcement
# ===========================================================================

@pytest.mark.parametrize("flag_value", ["1", "true", "True", "yes", "YES"])
def test_cache_offline_rejects_uncached_url(tmp_path, monkeypatch, flag_value):
    """HYPERGAN_OFFLINE rejection across all declared truthy variations with zero network requests."""
    cache_dir = tmp_path / "offline_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HYPERGAN_OFFLINE", flag_value)
    monkeypatch.setenv("HYPERGAN_CACHE_DIR", str(cache_dir))
    assert is_offline() is True

    with _AdversarialHTTPServer({"/weights.pth": b"never delivered"}) as server:
        url = server.url_for("/weights.pth")

        # 1. ensure_cached_url must raise OfflineError
        with pytest.raises(OfflineError) as exc_info:
            ensure_cached_url(url, cache_dir=cache_dir)
        assert issubclass(OfflineError, FileNotFoundError)
        assert "offline mode" in str(exc_info.value).lower()

        # 2. resolve_or_cache must also raise OfflineError
        with pytest.raises(OfflineError):
            resolve_or_cache(url, cache_dir=cache_dir)

        # 3. Assert zero network requests touched the server
        assert len(server.requests) == 0


def test_cache_offline_passes_cached_valid_file(tmp_path, monkeypatch):
    """HYPERGAN_OFFLINE allows hits on already-cached files without network calls."""
    cache_dir = tmp_path / "offline_hit"
    cache_dir.mkdir(parents=True, exist_ok=True)

    payload = b"OFFLINE_PRECACHED_ASSET_CONTENT"
    sha = hashlib.sha256(payload).hexdigest()
    (cache_dir / "valid_model.pth").write_bytes(payload)

    monkeypatch.setenv("HYPERGAN_OFFLINE", "1")
    monkeypatch.setenv("HYPERGAN_CACHE_DIR", str(cache_dir))

    with _AdversarialHTTPServer({"/valid_model.pth": b"network failover"}) as server:
        url = server.url_for("/valid_model.pth")
        resolved = resolve_or_cache(url, expected_sha256=sha, cache_dir=cache_dir)
        assert resolved == cache_dir / "valid_model.pth"
        assert resolved.read_bytes() == payload
        # Server must have received 0 requests
        assert len(server.requests) == 0


def test_cache_offline_purges_corrupted_file_and_raises_offline_error(tmp_path, monkeypatch):
    """If existing cached file has corrupt digest, offline mode removes it and raises OfflineError."""
    cache_dir = tmp_path / "offline_corrupt"
    cache_dir.mkdir(parents=True, exist_ok=True)

    tampered_file = cache_dir / "corrupted_weight.pth"
    tampered_file.write_bytes(b"BAD_CORRUPT_BYTES")

    expected_sha = hashlib.sha256(b"GENUINE_WEIGHTS").hexdigest()

    monkeypatch.setenv("HYPERGAN_OFFLINE", "1")
    monkeypatch.setenv("HYPERGAN_CACHE_DIR", str(cache_dir))

    with _AdversarialHTTPServer({"/corrupted_weight.pth": b"GENUINE_WEIGHTS"}) as server:
        url = server.url_for("/corrupted_weight.pth")
        with pytest.raises(OfflineError, match="offline mode"):
            resolve_or_cache(url, expected_sha256=expected_sha, cache_dir=cache_dir)

        # Corrupted file must have been purged from disk
        assert not tampered_file.exists()
        # Zero network requests permitted
        assert len(server.requests) == 0


def test_cache_offline_allows_local_paths(tmp_path, monkeypatch):
    """HYPERGAN_OFFLINE does not prevent resolving local files."""
    monkeypatch.setenv("HYPERGAN_OFFLINE", "1")
    local_file = tmp_path / "local_weights.pth"
    payload = b"LOCAL_UNTOUCHED_BYTES"
    local_file.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()

    resolved = resolve_or_cache(local_file, expected_sha256=sha)
    assert resolved == local_file.resolve()


# ===========================================================================
# 3. Cache Hash Corruption & Partial Resume
# ===========================================================================

def test_cache_sha256_mismatch_raises_and_cleans_up(tmp_path):
    """Server sends incorrect data: verify ValueError raised and .part file cleaned up."""
    cache_dir = tmp_path / "cache_sha_err"
    payload = b"ACTUAL_DELIVERED_BYTES"
    expected_sha = "f" * 64

    with _AdversarialHTTPServer({"/mismatch.bin": payload}) as server:
        url = server.url_for("/mismatch.bin")
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            ensure_cached_url(url, expected_sha256=expected_sha, cache_dir=cache_dir)

        assert not (cache_dir / "mismatch.bin").exists()
        assert not (cache_dir / "mismatch.bin.part").exists()


def test_cache_download_exceeding_declared_size_aborts(tmp_path):
    """Stream that exceeds declared expected_size aborts mid-flight and unlinks .part."""
    cache_dir = tmp_path / "cache_size_err"
    payload = b"A" * 1024

    with _AdversarialHTTPServer({"/oversized.bin": payload}) as server:
        url = server.url_for("/oversized.bin")
        # Declare size as only 100 bytes
        with pytest.raises(ValueError, match="exceeded declared size"):
            ensure_cached_url(url, expected_size=100, cache_dir=cache_dir)

        assert not (cache_dir / "oversized.bin").exists()
        assert not (cache_dir / "oversized.bin.part").exists()


def test_cache_partial_range_resume_success(tmp_path):
    """Interrupted download resumes cleanly via HTTP Range (206 Partial Content)."""
    cache_dir = tmp_path / "cache_resume"
    cache_dir.mkdir(parents=True, exist_ok=True)

    payload = b"STREAMING_RESUME_CHUNKS_0123456789_" * 2000  # 70 KB
    sha = hashlib.sha256(payload).hexdigest()
    split_point = 25000

    # Simulate existing partial file from aborted download
    part_file = cache_dir / "resumable.bin.part"
    part_file.write_bytes(payload[:split_point])
    assert part_file.stat().st_size == split_point

    with _AdversarialHTTPServer({"/resumable.bin": payload}, support_range=True) as server:
        url = server.url_for("/resumable.bin")
        result = ensure_cached_url(url, expected_sha256=sha, expected_size=len(payload), cache_dir=cache_dir)

        assert result == cache_dir / "resumable.bin"
        assert result.read_bytes() == payload
        assert not part_file.exists()

        assert len(server.requests) == 1
        assert server.requests[0]["range"] == f"bytes={split_point}-"


def test_cache_oversized_part_restarts_cleanly(tmp_path):
    """If existing .part file is already larger than expected_size, reset to byte 0."""
    cache_dir = tmp_path / "cache_oversized_part"
    cache_dir.mkdir(parents=True, exist_ok=True)

    payload = b"FRESH_DOWNLOAD_CONTENT"
    sha = hashlib.sha256(payload).hexdigest()

    part_file = cache_dir / "reset.bin.part"
    part_file.write_bytes(b"X" * (len(payload) + 1000))  # Stale oversized part

    with _AdversarialHTTPServer({"/reset.bin": payload}) as server:
        url = server.url_for("/reset.bin")
        result = ensure_cached_url(url, expected_sha256=sha, expected_size=len(payload), cache_dir=cache_dir)

        assert result == cache_dir / "reset.bin"
        assert result.read_bytes() == payload
        assert not part_file.exists()


def test_cache_range_416_resets_and_retries_cleanly(tmp_path):
    """If server responds with HTTP 416 (Range Not Satisfiable), cache unlinks .part and retries from 0."""
    cache_dir = tmp_path / "cache_416"
    cache_dir.mkdir(parents=True, exist_ok=True)

    payload = b"RECOVERED_FROM_416_PAYLOAD"
    sha = hashlib.sha256(payload).hexdigest()

    # Create .part with an invalid offset that triggers 416
    part_file = cache_dir / "retry_416.bin.part"
    part_file.write_bytes(b"GARBAGE" * 1000)

    with _AdversarialHTTPServer({"/retry_416.bin": payload}, support_range=True) as server:
        url = server.url_for("/retry_416.bin")
        result = ensure_cached_url(url, expected_sha256=sha, cache_dir=cache_dir)

        assert result == cache_dir / "retry_416.bin"
        assert result.read_bytes() == payload
        assert not part_file.exists()
        # Verify 2 requests: initial failed range request + retried full request
        assert len(server.requests) == 2


def test_cache_corrupted_existing_file_healed(tmp_path):
    """When an existing cached target is corrupted, ensure_cached_url re-downloads and heals it."""
    cache_dir = tmp_path / "cache_heal"
    cache_dir.mkdir(parents=True, exist_ok=True)

    genuine_payload = b"GENUINE_HEALED_DATA"
    genuine_sha = hashlib.sha256(genuine_payload).hexdigest()

    target = cache_dir / "healed.bin"
    target.write_bytes(b"CORRUPTED_DISC_BITROT")

    with _AdversarialHTTPServer({"/healed.bin": genuine_payload}) as server:
        url = server.url_for("/healed.bin")
        result = ensure_cached_url(url, expected_sha256=genuine_sha, cache_dir=cache_dir)

        assert result == target
        assert result.read_bytes() == genuine_payload


# ===========================================================================
# 4. Cache Local Path Rules
# ===========================================================================

def test_cache_local_path_strict_no_copy_or_modify(tmp_path):
    """Local path pass-through strictly avoids creating or touching files in cache dir."""
    cache_dir = tmp_path / "isolated_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_model = tmp_path / "my_local_model.pt"
    local_data = b"STRICT_LOCAL_WEIGHTS_DO_NOT_COPY"
    local_model.write_bytes(local_data)
    local_sha = hashlib.sha256(local_data).hexdigest()

    # 1. Standard filesystem path
    res = resolve_or_cache(local_model, expected_sha256=local_sha, cache_dir=cache_dir)
    assert res == local_model.resolve()
    assert list(cache_dir.iterdir()) == []

    # 2. String representation
    res_str = resolve_or_cache(str(local_model), expected_sha256=local_sha, cache_dir=cache_dir)
    assert res_str == local_model.resolve()
    assert list(cache_dir.iterdir()) == []

    # 3. file:// URI
    file_uri = f"file://{local_model.resolve()}"
    res_uri = resolve_or_cache(file_uri, expected_sha256=local_sha, cache_dir=cache_dir)
    assert res_uri == local_model.resolve()
    assert list(cache_dir.iterdir()) == []


def test_cache_local_path_sha_mismatch_raises_without_unlinking(tmp_path):
    """Local path SHA mismatch raises ValueError, and NEVER deletes user local file."""
    local_file = tmp_path / "user_dataset.bin"
    local_file.write_bytes(b"ORIGINAL_USER_DATA")
    wrong_sha = "0" * 64

    with pytest.raises(ValueError, match="Local artifact SHA256 mismatch"):
        resolve_or_cache(local_file, expected_sha256=wrong_sha)

    # File must NOT be unlinked
    assert local_file.is_file()
    assert local_file.read_bytes() == b"ORIGINAL_USER_DATA"


def test_cache_missing_local_file_raises_filenotfound(tmp_path):
    """Missing local file raises FileNotFoundError without touching network or cache."""
    cache_dir = tmp_path / "cache_empty"
    cache_dir.mkdir(parents=True, exist_ok=True)

    missing_path = tmp_path / "missing_file.pt"
    with pytest.raises(FileNotFoundError, match="missing"):
        resolve_or_cache(missing_path, cache_dir=cache_dir)

    assert list(cache_dir.iterdir()) == []


# ===========================================================================
# 5. Queue SIGTERM Forwarding
# ===========================================================================

def test_queue_sigterm_forwards_to_child_and_terminates(tmp_path):
    """SIGTERM sent to queue_run process forwards SIGINT to child and cleanly exits."""
    q_dir = tmp_path / ".hypergan"
    q_dir.mkdir(parents=True)
    q_file = q_dir / "queue.json"

    model_file = tmp_path / "model.toml"
    model_file.write_text('format = "hypergan-model/1"\n')

    queue_add(model_file, queue_file=q_file)

    # Runner script that runs queue_run with a mock child trainer that catches SIGINT and exits with 42
    runner_code = f"""
import sys, time, subprocess, signal
import hypergan.queue as qmod

orig_popen = subprocess.Popen
def mock_popen(cmd, **kwargs):
    sub_cmd = [
        sys.executable,
        "-c",
        "import time, signal, sys; signal.signal(signal.SIGINT, lambda s, f: sys.exit(42)); time.sleep(30)"
    ]
    return orig_popen(sub_cmd, **kwargs)

qmod.subprocess.Popen = mock_popen

ret = qmod.queue_run(queue_file="{q_file}", once=True)
sys.exit(ret)
"""

    runner_script = tmp_path / "runner_test.py"
    runner_script.write_text(runner_code)

    runner_proc = subprocess.Popen([sys.executable, str(runner_script)])

    # Poll until child is running
    child_pid = None
    for _ in range(50):
        time.sleep(0.1)
        st = queue_status(queue_file=q_file)
        active = st.get("active_job")
        if active and active.get("status") == "running" and active.get("pid"):
            child_pid = active["pid"]
            break

    assert child_pid is not None, "Child process failed to start"
    os.kill(child_pid, 0)  # Verify alive

    # Deliver SIGTERM to queue runner
    runner_proc.send_signal(signal.SIGTERM)

    runner_proc.wait(timeout=6.0)
    assert runner_proc.returncode == 0

    # Verify child process terminated
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)

    # Verify queue status updated
    final_st = queue_status(queue_file=q_file)
    final_job = final_st.get("active_job")
    assert final_job["pid"] is None
    assert final_job["exit_code"] == 42
    assert final_job["status"] == "failed"


def test_queue_sigterm_kills_unresponsive_child_via_fallback(tmp_path):
    """If child ignores SIGINT, runner falls back to proc.kill() to ensure child termination."""
    q_dir = tmp_path / ".hypergan"
    q_dir.mkdir(parents=True)
    q_file = q_dir / "queue.json"

    model_file = tmp_path / "model.toml"
    model_file.write_text('format = "hypergan-model/1"\n')

    queue_add(model_file, queue_file=q_file)

    runner_code = f"""
import sys, time, subprocess, signal
import hypergan.queue as qmod

orig_popen = subprocess.Popen
def mock_popen(cmd, **kwargs):
    # Child ignores SIGINT completely
    sub_cmd = [
        sys.executable,
        "-c",
        "import time, signal; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(60)"
    ]
    proc = orig_popen(sub_cmd, **kwargs)
    # Monkeypatch the 10.0s wait timeout to 1.0s for test speed
    orig_wait = proc.wait
    def fast_wait(timeout=None):
        if timeout == 10.0:
            timeout = 1.0
        return orig_wait(timeout=timeout)
    proc.wait = fast_wait
    return proc

qmod.subprocess.Popen = mock_popen

ret = qmod.queue_run(queue_file="{q_file}", once=True)
sys.exit(ret)
"""

    runner_script = tmp_path / "runner_unkillable.py"
    runner_script.write_text(runner_code)

    runner_proc = subprocess.Popen([sys.executable, str(runner_script)])

    child_pid = None
    for _ in range(50):
        time.sleep(0.1)
        st = queue_status(queue_file=q_file)
        active = st.get("active_job")
        if active and active.get("status") == "running" and active.get("pid"):
            child_pid = active["pid"]
            break

    assert child_pid is not None
    os.kill(child_pid, 0)

    # Send SIGTERM to runner
    runner_proc.send_signal(signal.SIGTERM)

    runner_proc.wait(timeout=6.0)
    assert runner_proc.returncode == 0

    # Verify unresponsive child was slain with SIGKILL
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)

    final_st = queue_status(queue_file=q_file)
    final_job = final_st.get("active_job")
    assert final_job["pid"] is None
    assert final_job["status"] == "failed"
    assert final_job["exit_code"] in (-9, 137, -signal.SIGKILL)


def test_queue_sigterm_during_idle_poll(tmp_path):
    """Queue runner polling an empty queue halts within one poll interval upon SIGTERM (PEP 475)."""
    q_dir = tmp_path / ".hypergan"
    q_dir.mkdir(parents=True)
    q_file = q_dir / "queue.json"

    runner_code = f"""
import sys
import hypergan.queue as qmod

ret = qmod.queue_run(queue_file="{q_file}", poll_interval=1.0)
sys.exit(ret)
"""
    runner_script = tmp_path / "runner_idle.py"
    runner_script.write_text(runner_code)

    proc = subprocess.Popen([sys.executable, str(runner_script)])
    time.sleep(0.3)

    # Deliver SIGTERM
    start_time = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=3.0)
    elapsed = time.monotonic() - start_time

    assert proc.returncode == 0
    assert elapsed < 2.5, f"Idle queue runner took too long to halt ({elapsed:.2f}s)"


# ===========================================================================
# 6. Checkpoints Pruning Race Stress
# ===========================================================================

def test_checkpoints_prune_concurrent_multiprocess_race(tmp_path):
    """Concurrent multi-process prune_checkpoints execution with lock verification and invariant safety."""
    run_dir = tmp_path / "prune_run"
    ckpt_root = run_dir / "checkpoints"
    ckpt_root.mkdir(parents=True)

    # Create 8 dummy checkpoints: steps 0, 100, 200, 300, 400, 500, 600, 700
    steps = [0, 100, 200, 300, 400, 500, 600, 700]
    ckpts = {}
    for s in steps:
        d = ckpt_root / f"att-1-step-{s:08d}-uuid{s}"
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"step": s, "attempt_id": "att-1"}), encoding="utf-8")
        (d / "state.pt").write_bytes(b"W" * 2048)
        ckpts[s] = d

    # latest.json points to step 700
    (ckpt_root / "latest.json").write_text(
        json.dumps({"schema_version": 1, "checkpoint": ckpts[700].name, "step": 700}),
        encoding="utf-8",
    )

    # Policy: keep_last=2 (600, 700), keep_every=500 (0, 500)
    # Expected retained: 0 (step_zero), 500 (milestone), 600 (keep_last), 700 (latest & keep_last)
    # Expected deleted: 100, 200, 300, 400
    policy = {"keep_last": 2, "keep_every": 500}

    ctx = mp.get_context("spawn")
    q = ctx.Queue()

    num_workers = 5
    procs = [
        ctx.Process(target=_mp_prune_worker, args=(str(run_dir), policy, q))
        for _ in range(num_workers)
    ]

    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=10)
        assert not p.is_alive()

    results = [q.get(timeout=2) for _ in range(num_workers)]

    # All workers must succeed
    for r in results:
        assert r["ok"] is True, f"Prune worker failed: {r}"

    # Across all workers, union of deleted steps must match exactly
    all_deleted = set()
    for r in results:
        all_deleted.update(r["deleted"])

    assert all_deleted == {100, 200, 300, 400}

    # Verify on-disk checkpoints
    remaining = [
        c["step"] for c in _durable_checkpoints(ckpt_root)
    ]
    assert remaining == [0, 500, 600, 700]

    for s in [0, 500, 600, 700]:
        assert ckpts[s].is_dir()
    for s in [100, 200, 300, 400]:
        assert not ckpts[s].exists()


def test_checkpoints_prune_lock_mutual_exclusion(tmp_path):
    """Verify _prune_lock blocks concurrent prune operations until released."""
    run_dir = tmp_path / "lock_run"
    ckpt_root = run_dir / "checkpoints"
    ckpt_root.mkdir(parents=True)

    for s in [0, 100, 200]:
        d = ckpt_root / f"att-1-step-{s:08d}-uuid{s}"
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"step": s, "attempt_id": "att-1"}), encoding="utf-8")
        (d / "state.pt").write_bytes(b"L" * 1024)

    (ckpt_root / "latest.json").write_text(
        json.dumps({"schema_version": 1, "checkpoint": f"att-1-step-00000200-uuid200", "step": 200}),
        encoding="utf-8",
    )

    events = []

    def background_prune():
        events.append("bg_started")
        rep = prune_checkpoints(run_dir, policy={"keep_last": 1, "keep_every": 0})
        events.append("bg_finished")
        return rep

    # Acquire lock in main thread
    with _prune_lock(ckpt_root):
        events.append("lock_held")
        t = threading.Thread(target=background_prune)
        t.start()
        time.sleep(0.3)
        # Background thread should be blocked waiting for lock
        assert "bg_started" in events
        assert "bg_finished" not in events
        events.append("lock_releasing")

    # Lock is now released
    t.join(timeout=3)
    assert "bg_finished" in events
    assert events == ["lock_held", "bg_started", "lock_releasing", "bg_finished"]


def test_checkpoints_prune_ignores_prune_lock_and_pending(tmp_path):
    """Ensure .prune.lock and .pending-* directories are never mistaken for durable checkpoints."""
    run_dir = tmp_path / "pending_run"
    ckpt_root = run_dir / "checkpoints"
    ckpt_root.mkdir(parents=True)

    # 1. Valid checkpoint
    c0 = ckpt_root / "att-1-step-00000000-uuid0"
    c0.mkdir()
    (c0 / "manifest.json").write_text(json.dumps({"step": 0, "attempt_id": "att-1"}), encoding="utf-8")
    (c0 / "state.pt").write_bytes(b"0" * 100)

    # 2. Lock file
    (ckpt_root / ".prune.lock").write_bytes(b"")

    # 3. Pending checkpoint write
    pending_dir = ckpt_root / ".pending-att-1-step-00000050-deadbeef1234"
    pending_dir.mkdir()
    (pending_dir / "manifest.json").write_text(json.dumps({"step": 50}), encoding="utf-8")

    durable = _durable_checkpoints(ckpt_root)
    assert len(durable) == 1
    assert durable[0]["step"] == 0
    assert durable[0]["name"] == c0.name

    report = prune_checkpoints(run_dir)
    assert report["retained_count"] == 1
    assert report["deleted_count"] == 0
    assert (ckpt_root / ".prune.lock").is_file()
