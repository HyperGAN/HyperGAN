"""
Opens a window that displays an image.
Usage:

    from viewer import GlobalViewer
    GlobalViewer.update(image)

"""
import numpy as np

class PygameViewer:

    def __init__(self, title="HyperGAN", enabled=True, zoom=1):
        self.screen = None
        self.title = title
        self.enabled = enabled
        self.zoom = zoom

    def update(self, image):
        if not self.enabled: return

        image = np.transpose(image, [1, 0,2])
        size = [image.shape[0], image.shape[1]]
        scaled = [image.shape[0]*self.zoom, image.shape[1]*self.zoom]
        if not self.screen:
            import pygame
            self.pg = pygame
            self.screen = self.pg.display.set_mode(scaled)
            self.pg.display.set_caption(self.title)
        self.pg.event.get()
        surface = self.pg.Surface(size)
        self.pg.surfarray.blit_array(surface, image)
        if self.zoom > 1:
            surface = self.pg.transform.scale(surface, scaled)
        self.screen.blit(surface, (0,0))
        self.pg.display.flip()

