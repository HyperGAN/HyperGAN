"""
Opens a window that displays an image.
Usage:

    from viewer import GlobalViewer
    GlobalViewer.update(image)

"""
import numpy as np

class PygameViewer:

    def __init__(self, title="HyperGAN", enabled=True):
        self.screen = None
        self.title = title
        self.enabled = enabled

    def update(self, image):
        if not self.enabled: return

        image = np.transpose(image, [1, 0,2])
        size = [image.shape[0], image.shape[1]]
        if not self.screen:
            import pygame
            self.pg = pygame
            self.screen = self.pg.display.set_mode(size)
            self.pg.display.set_caption(self.title)
        surface = self.pg.Surface(size)
        self.pg.surfarray.blit_array(surface, image)
        self.screen.blit(surface, (0,0))
        self.pg.display.flip()

