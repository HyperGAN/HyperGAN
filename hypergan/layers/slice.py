import torch.nn as nn
import hypergan as hg
from hypergan.layer_shape import LayerShape

class Slice(hg.Layer):
    def __init__(self, component, args, options):
        super(Slice, self).__init__(component, args, options)
        self.dim = options.dim or 1

        dims = list(component.current_size.dims)
        dims[0] = options.c or dims[0]
        dims[1] = options.h or dims[1]
        dims[2] = options.w or dims[2]
        self.size = LayerShape(*dims)

    def forward(self, input, context):

        return input[:, :self.size.dims[0], :self.size.dims[1], :self.size.dims[2]]

    def output_size(self):
        return self.size

