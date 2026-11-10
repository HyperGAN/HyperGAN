"""HyperGAN's lightweight public package.

Training dependencies are loaded only when a training or sampling API is used.
"""
from typing import TYPE_CHECKING

__version__ = "2.0.0a3"

if TYPE_CHECKING:
    from .api import (
        ConfigError,
        LoadedGenerator,
        Model,
        ModelDescription,
        ModelFileError,
        PsSnapshot,
        Run,
        RunHandle,
        RunInfo,
        RunStatus,
        Sample,
        SampleResult,
        StopTraining,
        TrainResult,
        ValidationWarning,
        compare,
        describe,
        events,
        export,
        interpolate,
        launch,
        load,
        load_bundle,
        logs,
        lower,
        metrics,
        new,
        operators,
        override,
        prune_checkpoints,
        ps,
        read_run_info,
        resolve,
        resume,
        rm,
        sample,
        samples,
        save,
        schema,
        status,
        stop,
        tag,
        train,
        validate,
        view,
    )

from .api import __all__ as _api_all, ALIASES as _ALIASES

__all__ = tuple(_api_all)
del _api_all


def __getattr__(name: str):
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    # The metrics module is callable as hg.metrics(run) and provides module attributes
    if name == "metrics":
        return importlib.import_module(".metrics", __name__)

    # Guard: only allow public symbols declared in __all__ (or their aliases)
    if name in __all__ or name in _ALIASES:
        api = importlib.import_module(".api", __name__)
        if hasattr(api, name):
            return getattr(api, name)

    # Allow direct submodule access (e.g. hypergan.cli, hypergan.config)
    try:
        return importlib.import_module(f".{name}", __name__)
    except ImportError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    # Return __all__ plus standard module dunder attributes present in globals
    dunders = {k for k in globals() if k.startswith("__") and k.endswith("__")}
    return sorted(dunders | set(__all__))
