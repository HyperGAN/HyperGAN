import torch.nn as nn
import hypergan as hg
from hypergan.layer_shape import LayerShape

class Expand(hg.Layer):
    def __init__(self, component, args, options):
        super(Expand, self).__init__(component, args, options)
        self.dim = options.dim or 1

        dims = list(component.current_size.dims)
        shape = [int(x) for x in str(args[0]).split("*")]
        dims = [dims[0]] + list(reversed(shape))
        self.size = LayerShape(*dims)

    def forward(self, input, context):
        expanded =  input.expand(-1, -1, self.size.dims[2], self.size.dims[1])
        return expanded

    def output_size(self):
        return self.size


