"""
Opens a window that displays an image.
Usage:

    from viewer import GlobalViewer
    GlobalViewer.update(image)

"""
import numpy as np



class PygameViewer:

    def __init__(self, title="HyperGAN", viewer_size=1, enabled=True):
        self.screen = None
        self.title = title
        self.viewer_size = viewer_size
        self.enabled = enabled

    def update(self, image):
        if not self.enabled: return

        image = np.transpose(image, [1, 0,2])

        if not self.screen:
            import pygame
            self.pg = pygame
            if self.viewer_size <= 0:
                self.viewer_size = 0.1
            self.size = [int(image.shape[0] * self.viewer_size), int(image.shape[1] * self.viewer_size)]
            self.aspect_w = image.shape[1] / image.shape[0]
            self.aspect_h = image.shape[0] / image.shape[1]
            self.temp_size = self.size
            self.screen = self.pg.display.set_mode(self.size,self.pg.RESIZABLE)
            self.pg.display.set_caption(self.title)

        for event in self.pg.event.get():
            if event.type == self.pg.VIDEORESIZE:
                if self.size[0] != event.size[0]:
                    self.temp_size = [event.size[0], int(event.size[0] * self.aspect_w)]
                elif self.size[1] != event.size[1]:
                    self.temp_size = [int(event.size[1] * self.aspect_h), event.size[1]]

            elif event.type == self.pg.ACTIVEEVENT and event.state == 2 and event.gain == 1:
                self.size = self.temp_size
                self.screen = self.pg.display.set_mode(self.size,self.pg.RESIZABLE)   


        surface = self.pg.Surface([image.shape[0],image.shape[1]])
        self.pg.surfarray.blit_array(surface, image)
        self.screen.blit(self.pg.transform.scale(surface,self.size),(0,0))
        self.pg.display.flip()

       
