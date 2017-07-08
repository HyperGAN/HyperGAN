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

    def __init__(self):
        self.screen = None

    def update(self, image):
        image = np.transpose(image, [1, 0,2])
        size = [image.shape[0], image.shape[1]]
        if not self.screen:
            self.screen = pygame.display.set_mode(size)
        surface = pygame.Surface(size)
        pygame.surfarray.blit_array(surface, image)
        self.screen.blit(surface, (0,0))
        pygame.display.flip()

