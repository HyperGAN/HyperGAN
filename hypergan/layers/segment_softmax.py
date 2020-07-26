import torch.nn as nn
import torch
from hypergan.layer_size import LayerSize
import hypergan as hg

class SegmentSoftmax(hg.Layer):
    def __init__(self, component, args, options):
        super(SegmentSoftmax, self).__init__(component, args, options)
        self.channels = args[0]
        self.dims = list(component.current_size.dims).copy()
        self.softmax = nn.Softmax(dim=2)

    def output_size(self):
        return LayerSize(*([self.channels]+self.dims[1:]))

    def forward(self, input, context):
        content, segment = torch.split(input, input.shape[1]//2, 1)
        net_in = content.view(content.shape[0], content.shape[1]//self.channels, self.channels, content.shape[2], content.shape[3])
        segment = segment.view(content.shape[0], content.shape[1]//self.channels, self.channels, content.shape[2], content.shape[3])
        selection = self.softmax(segment)
        rendered = (selection * net_in).sum(dim=1).view([input.shape[0]]+list(self.output_size().dims))
        return rendered
