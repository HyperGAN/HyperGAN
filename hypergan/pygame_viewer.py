"""
Opens a window that displays an image.
Usage:

    from viewer import GlobalViewer
    GlobalViewer.update(image)

Delays loading Gtk and friends until enable() is called.
"""
import numpy as np
import pygame

class PygameViewer:

    def __init__(self, title="HyperGAN"):
        self.screen = None
        self.title = title

    def update(self, image):
        image = np.transpose(image, [1, 0,2])
        size = [image.shape[0], image.shape[1]]
        if not self.screen:
            self.screen = pygame.display.set_mode(size)
            pygame.display.set_caption(self.title)
        surface = pygame.Surface(size)
        pygame.surfarray.blit_array(surface, image)
        self.screen.blit(surface, (0,0))
        pygame.display.flip()

