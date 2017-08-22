"""
Opens a window that displays an image.
Usage:

    from viewer import GlobalViewer
    GlobalViewer.update(image)

Delays loading Gtk and friends until enable() is called.
"""
import numpy as np

from hypergan.pygame_viewer import PygameViewer

GlobalViewer = PygameViewer()
