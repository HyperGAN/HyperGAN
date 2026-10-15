"""HyperGAN's lightweight public package.

Training dependencies are loaded only when a training or sampling API is used.
"""

__version__ = "2.0.0a1"

from .workspace import find_workspace, load_workspace, interpolate_variables, resolve_device_aliases

