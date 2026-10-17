"""HyperGAN's lightweight public package.

Training dependencies are loaded only when a training or sampling API is used.
"""

__version__ = "2.0.0a1"

from .workspace import find_workspace, load_workspace, interpolate_variables, resolve_device_aliases


def __getattr__(name: str):
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    api = importlib.import_module(".api", __name__)
    if name == "api":
        return api
    if hasattr(api, name):
        return getattr(api, name)
    try:
        return importlib.import_module(f".{name}", __name__)
    except ImportError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None



def __dir__() -> list[str]:
    import importlib

    api = importlib.import_module(".api", __name__)
    return sorted(set(globals()) | set(dir(api)))


