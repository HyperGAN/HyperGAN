"""
Opens a window that displays an image.
Usage:

    from viewer import GlobalViewer
    GlobalViewer.update(image)

Delays loading Gtk and friends until enable() is called.
"""
from hypergan.pygame_viewer import PygameViewer
from hypergan.tk_viewer import TkViewer

GlobalViewer = TkViewer()
