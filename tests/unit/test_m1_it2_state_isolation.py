"""Adversarial Stress Test Suite for Milestone 1 Iteration 2 (State Isolation & Conftest Remediation).

Scope:
1. tmp_path Purity & Decoupling:
   - Empirically verify that `isolate_hypergan_state` never injects directories or files into `tmp_path`.
   - Stress-test `assert list(tmp_path.iterdir()) == []` across parameterized tests and fixture boundaries.
2. Exception Resilience & Environment Variable Restoration:
   - Empirically verify that `HYPERGAN_STATE_DIR` is reliably restored even if test bodies or nested fixtures raise exceptions.
   - Verify behavior when previous environment was unset vs pre-set.
3. Host Registry Complete Immutability:
   - Empirically verify that `~/.local/share/hypergan/runs.json` on the host remains 100% untouched (SHA-256, size, mtime)
     under heavy concurrent registry read, write, and lock operations.
4. Concurrency & Multi-worker State Safety:
   - Stress-test concurrent processes and threads interacting with registry under isolated state directories.
"""
import concurrent.futures
import hashlib
import os
from pathlib import Path
import shutil
import sys
import threading
import time

import pytest

from hypergan.registry import (
    file_lock,
    get_registry_dir,
    list_registered_runs,
    register_run,
    unregister_run,
)
from tests.conftest import isolate_hypergan_state


# ==============================================================================
# 1. tmp_path Purity & Decoupling Verification
# ==============================================================================

def test_tmp_path_is_pristine_on_entry(tmp_path):
    """Verify that tmp_path contains zero files or subdirectories on test entry."""
    assert list(tmp_path.iterdir()) == [], f"tmp_path polluted with: {list(tmp_path.iterdir())}"


@pytest.mark.parametrize("iteration", range(15))
def test_tmp_path_purity_across_parameterized_invocations(tmp_path, iteration):
    """Stress-test that parameterized repetitions never cross-contaminate or leave state in tmp_path."""
    assert list(tmp_path.iterdir()) == [], f"Iteration {iteration} encountered dirty tmp_path: {list(tmp_path.iterdir())}"
    state_dir = Path(os.environ["HYPERGAN_STATE_DIR"])
    assert state_dir.is_dir(), f"State directory {state_dir} does not exist"
    # State dir must not be inside tmp_path
    assert not str(state_dir.resolve()).startswith(str(tmp_path.resolve())), "state_dir is inside tmp_path!"
    assert not str(tmp_path.resolve()).startswith(str(state_dir.resolve())), "tmp_path is inside state_dir!"


def test_state_dir_path_disjointness(tmp_path):
    """Empirically prove state_dir and tmp_path are completely disjoint directory trees."""
    state_dir = Path(os.environ["HYPERGAN_STATE_DIR"]).resolve()
    test_tmp = tmp_path.resolve()

    assert state_dir != test_tmp
    assert state_dir not in test_tmp.parents
    assert test_tmp not in state_dir.parents
    # tmp_path must be empty
    assert list(test_tmp.iterdir()) == []


def test_registry_operations_do_not_leak_into_tmp_path(tmp_path):
    """Perform actual registry mutations and verify tmp_path is untouched."""
    assert list(tmp_path.iterdir()) == []

    # Run directory inside a dedicated subfolder of tmp_path
    my_run = tmp_path / "sample_run"
    my_run.mkdir(parents=True, exist_ok=True)
    manifest = my_run / "manifest.json"
    manifest.write_text('{"schema": 1, "created_at": "2026-10-30T00:00:00Z"}')

    # Register the run
    register_run(my_run, name="adversarial_run_1")

    # The only thing in tmp_path should be sample_run
    entries = list(tmp_path.iterdir())
    assert entries == [my_run], f"Unexpected entries in tmp_path: {entries}"

    # Registry state must be in HYPERGAN_STATE_DIR
    reg_dir = get_registry_dir()
    assert reg_dir == Path(os.environ["HYPERGAN_STATE_DIR"]).resolve()
    assert (reg_dir / "runs.json").is_file()

    # List runs and verify registration succeeded in isolated state
    runs = list_registered_runs()
    assert any(r.get("name") == "adversarial_run_1" for r in runs)


# ==============================================================================
# 2. Environment Restoration & Exception Resilience
# ==============================================================================

def test_isolate_hypergan_state_restores_env_on_unhandled_exception(tmp_path_factory):
    """Empirically verify generator fixture cleans up when test body raises an exception."""
    fixture_func = isolate_hypergan_state.__wrapped__

    # Case A: When HYPERGAN_STATE_DIR was originally NOT set
    orig_env = os.environ.get("HYPERGAN_STATE_DIR")
    if orig_env is not None:
        del os.environ["HYPERGAN_STATE_DIR"]

    gen_unset = fixture_func(tmp_path_factory)
    yielded_state = next(gen_unset)
    assert Path(yielded_state).is_dir()
    assert os.environ.get("HYPERGAN_STATE_DIR") == str(yielded_state)

    # Simulate an unhandled exception thrown into the fixture
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        gen_unset.throw(ZeroDivisionError("division by zero"))

    # Environment variable MUST be deleted (not None string, completely popped)
    assert "HYPERGAN_STATE_DIR" not in os.environ

    # Case B: When HYPERGAN_STATE_DIR was originally set to a custom value
    custom_sentinel = "/custom/sentinel/path/to/hypergan/state"
    os.environ["HYPERGAN_STATE_DIR"] = custom_sentinel

    gen_set = fixture_func(tmp_path_factory)
    yielded_state_b = next(gen_set)
    assert Path(yielded_state_b).is_dir()
    assert os.environ.get("HYPERGAN_STATE_DIR") == str(yielded_state_b)
    assert yielded_state_b != custom_sentinel

    # Simulate an unhandled exception
    with pytest.raises(KeyError, match="synthetic_adversarial_error"):
        gen_set.throw(KeyError("synthetic_adversarial_error"))

    # Environment variable MUST be restored to custom_sentinel
    assert os.environ.get("HYPERGAN_STATE_DIR") == custom_sentinel

    # Restore environment for subsequent tests
    if orig_env is not None:
        os.environ["HYPERGAN_STATE_DIR"] = orig_env
    else:
        os.environ.pop("HYPERGAN_STATE_DIR", None)


def test_isolate_hypergan_state_restores_env_on_normal_exit(tmp_path_factory):
    """Empirically verify generator fixture cleans up on clean fixture exit."""
    fixture_func = isolate_hypergan_state.__wrapped__
    sentinel = "/test/normal/exit/state"
    os.environ["HYPERGAN_STATE_DIR"] = sentinel

    gen = fixture_func(tmp_path_factory)
    yielded = next(gen)
    assert os.environ["HYPERGAN_STATE_DIR"] == str(yielded)

    # Simulate normal completion
    with pytest.raises(StopIteration):
        next(gen)

    assert os.environ["HYPERGAN_STATE_DIR"] == sentinel
    os.environ.pop("HYPERGAN_STATE_DIR", None)


# ==============================================================================
# 3. Host Registry Immutability Invariant
# ==============================================================================

def test_host_runs_json_remains_strictly_untouched(tmp_path):
    """Empirically prove host ~/.local/share/hypergan/runs.json is 100% immutable."""
    host_runs = Path.home() / ".local" / "share" / "hypergan" / "runs.json"
    host_exists = host_runs.is_file()

    init_stat = host_runs.stat() if host_exists else None
    init_bytes = host_runs.read_bytes() if host_exists else None
    init_hash = hashlib.sha256(init_bytes).hexdigest() if host_exists else None

    # Perform extensive registry operations in this test
    state_dir = Path(os.environ["HYPERGAN_STATE_DIR"])
    for i in range(10):
        run_dir = tmp_path / f"run_immutability_{i}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text('{"schema": 1}')
        register_run(run_dir, name=f"immutability_test_{i}")

    # Check local isolated registry
    assert (state_dir / "runs.json").is_file()
    isolated_runs = list_registered_runs(cwd=tmp_path)
    assert len(isolated_runs) == 10

    # Unregister half of them and delete their directories
    for i in range(5):
        run_to_del = tmp_path / f"run_immutability_{i}"
        shutil.rmtree(run_to_del)
        unregister_run(run_to_del)
    assert len(list_registered_runs(cwd=tmp_path)) == 5

    # Verify host runs.json status
    if host_exists:
        curr_stat = host_runs.stat()
        curr_bytes = host_runs.read_bytes()
        curr_hash = hashlib.sha256(curr_bytes).hexdigest()

        assert curr_stat.st_size == init_stat.st_size, "Host runs.json size changed!"
        assert curr_stat.st_mtime == init_stat.st_mtime, "Host runs.json mtime was modified!"
        assert curr_hash == init_hash, "Host runs.json content hash changed!"
    else:
        assert not host_runs.exists(), "Host runs.json was created on host filesystem!"


# ==============================================================================
# 4. Multi-Worker Concurrency & Lock Stress Testing
# ==============================================================================

def _concurrent_worker_task(state_path: str, worker_id: int, num_runs: int) -> bool:
    """Worker process: sets HYPERGAN_STATE_DIR and performs multiple registrations and queries."""
    os.environ["HYPERGAN_STATE_DIR"] = state_path
    state = Path(state_path)
    state.mkdir(parents=True, exist_ok=True)

    for i in range(num_runs):
        run_path = state / f"run_{worker_id}_{i}"
        run_path.mkdir(parents=True, exist_ok=True)
        (run_path / "manifest.json").write_text(f'{{"schema": 1, "worker": {worker_id}, "run": {i}}}')
        register_run(run_path, name=f"worker_{worker_id}_run_{i}")

    runs = list_registered_runs()
    assert len(runs) >= num_runs
    return True


def test_concurrent_multiprocess_state_isolation(tmp_path):
    """Stress test 6 concurrent worker processes using isolated state directories."""
    host_runs = Path.home() / ".local" / "share" / "hypergan" / "runs.json"
    host_exists = host_runs.is_file()
    init_hash = hashlib.sha256(host_runs.read_bytes()).hexdigest() if host_exists else None

    workers = 6
    runs_per_worker = 5
    worker_states = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for w in range(workers):
            # Each worker gets its own isolated state directory
            w_state = str(tmp_path / f"worker_{w}_state")
            worker_states.append(Path(w_state))
            futures.append(executor.submit(_concurrent_worker_task, w_state, w, runs_per_worker))

        for f in concurrent.futures.as_completed(futures):
            assert f.result() is True

    # Verify each worker's state directory contains its own runs.json
    for w_state in worker_states:
        runs_file = w_state / "runs.json"
        assert runs_file.is_file(), f"Worker state {w_state} missing runs.json"

    # Verify host runs.json was NEVER touched
    if host_exists:
        curr_hash = hashlib.sha256(host_runs.read_bytes()).hexdigest()
        assert curr_hash == init_hash, "Host runs.json was modified during concurrent execution!"


def test_multithreaded_file_lock_isolation(tmp_path):
    """Stress test file_lock contention within isolated state directory."""
    state_dir = Path(os.environ["HYPERGAN_STATE_DIR"])
    lock_file_path = state_dir / "registry.lock"

    counter = {"value": 0}
    num_threads = 8
    increments_per_thread = 20

    def worker():
        for _ in range(increments_per_thread):
            with file_lock(lock_file_path, timeout=5.0):
                c = counter["value"]
                time.sleep(0.001)
                counter["value"] = c + 1

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter["value"] == num_threads * increments_per_thread
    assert list(tmp_path.iterdir()) == [], f"tmp_path was polluted by lock operations: {list(tmp_path.iterdir())}"
