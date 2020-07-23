import torch.nn as nn
from hypergan.layer_size import LayerSize
import hypergan as hg

class SegmentSoftmax(hg.Layer):
    def __init__(self, component, args, options):
        super(SegmentSoftmax, self).__init__(component, args, options)
        self.channels = args[0]
        self.dims = list(component.current_size.dims).copy()
        self.softmax = nn.Softmax(dim=1)

    def output_size(self):
        return LayerSize(self.channels, self.dims[1], self.dims[2])

    def forward(self, input, context):
        net_in = input.view(input.shape[0], input.shape[1] // self.channels, self.channels, input.shape[2], input.shape[3])
        selection = self.softmax(net_in)
        rendered = (selection * net_in).sum(dim=1)
        return rendered.view([input.shape[0]]+list(self.output_size().dims))
