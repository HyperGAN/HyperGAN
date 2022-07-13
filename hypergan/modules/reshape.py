import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, *dims):
        self.dims = dims
        super(Reshape, self).__init__()
    def forward(self, x):
        return x.reshape(x.size()[0], *self.dims)
