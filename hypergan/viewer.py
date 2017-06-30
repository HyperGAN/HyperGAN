"""
Opens a window that displays an image.
Usage:

    from viewer import GlobalViewer
    GlobalViewer.enable()
    GlobalViewer.update(image)

Delays loading Gtk and friends until enable() is called.
"""
import numpy as np

class Viewer:
    def __init__(self):
        self.started = False
        self.can_update = False

    def enable(self):
        if self.started: return
        self.started = True
        self.size = [0, 0]
        import gi
        gi.require_version('Gtk', '3.0')
        from gi.repository import GObject
        GObject.threads_init()
        from gi.repository import GLib
        from gi.repository import Gtk
        from gi.repository import Gdk
        from gi.repository import GdkPixbuf
        self.GLib = GLib
        self.Gtk = Gtk
        self.Gdk = Gdk
        self.GdkPixbuf = GdkPixbuf
        self.window = Gtk.Window()
        self.im = Gtk.Image()
        self.window.add(self.im)
        self.window.set_position(Gtk.WindowPosition.CENTER)
        self.window.set_decorated(True)
        # Hint for tiling window managers to float this window:
        self.window.set_type_hint(Gdk.WindowTypeHint.UTILITY)
        def on_delete(unused1=None, unused2=None):
            self.can_update = False
            Gtk.main_quit()
        self.window.connect('delete_event', on_delete)
        def on_key_press(widget_unused, event):
            key = Gdk.keyval_name(event.keyval)
            if key in ('Escape', 'q'):
                self.window.destroy()
                on_delete()
        self.window.connect("key_press_event", on_key_press)
        self.window.show_all()
        self.can_update = True
        import threading
        self.thread = threading.Thread(target=Gtk.main, name='gtkmain', daemon=True)
        self.thread.start()

    def update(self, image):
        if not self.can_update: return
        self.GLib.idle_add(self._do_update_on_gtk_thread, image)

    def _do_update_on_gtk_thread(self, image):
        h, w = image.shape[0:2]
        if len(image.shape) != 3:
            # Convert greyscale to RGB.
            old = image
            image = np.zeros((h, w, 3), dtype=np.uint8)
            image[:,:,:] = np.atleast_3d(old)[:,:]
        if self.size != [w, h]:
            self.size = [w, h]
            self.window.resize(w, h)
            self.window.move((self.Gdk.Screen.width() - w) / 2,
                             (self.Gdk.Screen.height() - h) / 2)
        have_alpha = False
        rowstride = w * 3
        # Avoid new_from_data: https://bugzilla.gnome.org/show_bug.cgi?id=721497
        self.Gdk.threads_enter()
        pixels = self.GLib.Bytes.new_take(image.tostring())
        self.pixbuf = self.GdkPixbuf.Pixbuf.new_from_bytes(pixels,
                self.GdkPixbuf.Colorspace.RGB,
                have_alpha, 8, w, h, rowstride)
        self.im.set_from_pixbuf(self.pixbuf.copy())
        self.im.show()
        self.Gdk.threads_leave()
        return False  # Don't call the same callback repeatedly.

GlobalViewer = Viewer()
