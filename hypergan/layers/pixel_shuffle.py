import torch.nn as nn
from hypergan.layer_size import LayerSize
import hypergan as hg

class PixelShuffle(hg.Layer):
    def __init__(self, component, args, options):
        super(PixelShuffle, self).__init__(component, args, options)
        self.shuffle = nn.PixelShuffle(2)
        self.dims = list(component.current_size.dims).copy()

    def output_size(self):
        return LayerSize(self.dims[0]//4, self.dims[1]*2, self.dims[2]*2)

    def forward(self, input, context):
        return self.shuffle(input)
