class LayerSize:
    def __init__(self, channels, height, width):
        self.channels = channels
        self.height = height
        self.width = width

    def size(self):
        return self.channels * self.height * self.width
