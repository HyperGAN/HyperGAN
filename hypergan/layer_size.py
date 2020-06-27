class LayerSize:
    def __init__(self, *dims):
        self.dims = dims
        if len(dims) == 1:
            self.channels = dims[0]
        elif len(dims) == 2:
            self.channels = dims[0]
            self.height = dims[1]
        elif len(dims) == 3:
            self.channels = dims[0]
            self.height = dims[1]
            self.width = dims[2]
        elif len(dims) == 4:
            self.frames = dims[0]
            self.channels = dims[1]
            self.height = dims[2]
            self.width = dims[3]

    def size(self):
        if len(self.dims) == 1:
            return self.channels
        if len(self.dims) == 2:
            return self.channels * self.height
        if len(self.dims) == 3:
            return self.channels * self.height * self.width
        return self.channels * self.height * self.width * self.frames
