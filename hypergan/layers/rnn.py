import torch.nn as nn
from hypergan.layer_shape import LayerShape
import hypergan as hg
import torch.nn.functional as F
import torch

class Rnn(hg.Layer):
    """
    """

    def __init__(self, component, args, options):
        super(Rnn, self).__init__(component, args, options)
        output_size = args[0]
        self.dims = [output_size]
        self.rnn = nn.RNN(component.current_size.height, output_size, options.num_layers or 2, bias=False)

    def output_size(self):
        return LayerShape(*self.dims)

    def forward(self, input, context):
        with torch.backends.cudnn.flags(enabled=False):
            output, h0 = self.rnn(input.permute(1,0,2))
            output = output[-1]

        return output
