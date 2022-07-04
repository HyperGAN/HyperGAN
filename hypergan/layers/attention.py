import torch
from hypergan.layer_shape import LayerShape
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as f
from hypergan.layer_shape import LayerShape
import hypergan as hg
from hypergan.modules.attention import Attention as AttentionModule
from hypergan.layers.cross_attention import Block

class Attention(hg.Layer):

    def __init__(self, component, args, options):
        super().__init__(component, args, options)
        in_channels = component.current_size.dims[0]
        channels = in_channels
        if len(args) > 0:
            channels = args[0]
        self.attention = AttentionModule(in_channels)
        self.block = Block(in_channels, channels, groups = 8)
        self.res_conv = nn.Conv2d(in_channels, channels, 1) if in_channels != channels else nn.Identity()
        self.size = LayerShape(channels, *component.current_size.dims[1:])
    def forward(self, x, context):
        attn = self.attention(x)
        res = self.res_conv(x)
        bl = self.block(attn) 
        return bl + res
    def output_size(self):
        return self.size
