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
        if len(dims) == 3:
            dims[2] = options.w or dims[2]
        self.size = LayerShape(*dims)

    def forward(self, input, context):
        c = self.size.dims[0] or input.shape[1]
        if len(input.shape) == 4:
            return input[:, :c, :self.size.dims[1], :self.size.dims[2]]
        if len(input.shape) == 3:
            return input[:, :c, :self.size.dims[1]]

    def output_size(self):
        return self.size

