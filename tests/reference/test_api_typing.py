"""Public API symbol resolution and training lifecycle with training dependencies."""
import hypergan as hg
from hypergan.api import RunHandle, StopTraining


def test_canonical_public_symbols_and_docstrings():
    """Resolve lazy training exports alongside the lightweight public API."""
    for name in hg.__all__:
        assert hasattr(hg, name), f"Symbol {name} in __all__ cannot be resolved on hg"
        obj = getattr(hg, name)
        if callable(obj) and not isinstance(obj, type):
            doc = getattr(obj, "__doc__", None)
            assert doc and doc.strip(), f"Public callable {name} is missing a docstring"


def test_early_stopping_with_stop_training_exception(tmp_path):
    """Verify on_event raising StopTraining triggers clean graceful early stop."""
    model = hg.packaged("gaussian-grid")
    run_dir = tmp_path / "stop_exc_run"

    step_count = 0

    def on_event(ev):
        nonlocal step_count
        if ev.get("event") == "train":
            step_count += 1
            if ev.get("step", 0) >= 20:
                raise StopTraining("Halted early by test callback")

    run = hg.train(model, run_dir=run_dir, steps=100, on_event=on_event)
    info = hg.read_run_info(run_dir)
    assert info.status == "stopped"
    assert "Halted early" in str(info.stop_reason)
    assert info.step >= 20
    assert info.step < 100


def test_early_stopping_with_return_false(tmp_path):
    """Verify on_event returning False triggers clean graceful early stop."""
    model = hg.packaged("gaussian-grid")
    run_dir = tmp_path / "return_false_run"

    def on_event(ev):
        if ev.get("event") == "train" and ev.get("step", 0) >= 15:
            return False
        return True

    run = hg.train(model, run_dir=run_dir, steps=100, on_event=on_event)
    info = hg.read_run_info(run_dir)
    assert info.status == "stopped"
    assert info.step >= 15
    assert info.step < 100


def test_launch_non_blocking_handle(tmp_path):
    """Verify hg.launch returns a RunHandle that runs asynchronously and can be waited on."""
    model = hg.packaged("gaussian-grid")
    run_dir = tmp_path / "launch_run"

    handle = hg.launch(model, run_dir=run_dir, steps=30)
    assert isinstance(handle, RunHandle)
    assert handle.is_alive() or handle.wait(timeout=10.0)
    final_run = handle.wait(timeout=15.0)
    assert final_run.path == run_dir
    assert not handle.is_alive()
    info = handle.run_info
    assert info.status in ("complete", "completed", "stopped")
    assert info.step >= 30
