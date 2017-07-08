"""
Opens a window that displays an image.
Usage:

    from viewer import GlobalViewer
    GlobalViewer.enable()
    GlobalViewer.update(image)

Delays loading Gtk and friends until enable() is called.
"""
import numpy as np
import pygame

class SdlViewer:
    def __init__(self, size = (2048, 512)):
        self.size = size

    def enable(self):
        self.screen = pygame.display.set_mode(self.size)

    def update(self, image):
        print("preImage is ",image.shape)
        image = np.transpose(image, [1, 0,2])
        print("Image is ",image.shape)
        surface = pygame.Surface(self.size)
        pygame.surfarray.blit_array(surface, image)
        #surface = pygame.transform.rotate(surface, 90)
        self.screen.blit(surface, (0,0))
        pygame.display.flip()

GlobalViewer = SdlViewer()
