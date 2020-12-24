"""
Opens a window that displays an image.
Usage:

    from viewer import GlobalViewer
    GlobalViewer.update(image)

Delays loading Gtk and friends until enable() is called.
"""
from hypergan.threaded_tk_viewer import ThreadedTkViewer

GlobalViewer = ThreadedTkViewer()
