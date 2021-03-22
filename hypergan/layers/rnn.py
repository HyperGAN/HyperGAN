import torch.nn as nn
from hypergan.layer_shape import LayerShape
import hypergan as hg
import torch.nn.functional as F

class Rnn(hg.Layer):
    """
    """

    def __init__(self, component, args, options):
        super(Rnn, self).__init__(component, args, options)
        self.dims = list(component.current_size.dims).copy()
        input_channels = self.dims[0]
        self.x2h = nn.Conv2d(input_channels, input_channels*3, 3, 1, padding = 1)
        self.h2h = nn.Conv2d(input_channels, input_channels*3, 3, 1, padding = 1)
        component.nn_init(self.x2h, options.initializer)
        component.nn_init(self.h2h, options.initializer)

    def output_size(self):
        return LayerShape(*self.dims)

    def forward(self, input, context):
        hidden = context["past"]

        x = input
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)
        return hy
