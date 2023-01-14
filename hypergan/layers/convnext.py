import torch
from torch import nn
from torch.nn import functional as f
from hypergan.layer_shape import LayerShape
import hypergan as hg
from timm.models.layers import trunc_normal_


class ConvNext(hg.Layer):
    def __init__(self, component, args, options):
        super(ConvNext, self).__init__(component, args, options)
        stacks = []
        depth = options.depth or 1
        self.size = LayerShape(*component.current_size.dims)
        in_dim = self.size.dims[0]
        out_dim = args[0]
        expansion = 4
        norm = nn.LayerNorm([out_dim, *self.size.dims[1:]])# if options.norm else nn.Identity()

        for i in range(depth):
            if options.deconv:
                stacks.append(nn.Sequential(
                    nn.ConvTranspose2d(in_dim, out_dim, 7, padding=3, groups=out_dim, output_padding=0),
                    norm,
                    nn.ConvTranspose2d(out_dim, out_dim*expansion, 1, padding=0),
                    nn.GELU(),
                    nn.ConvTranspose2d(out_dim*expansion, out_dim, 1, padding=0)
                ))

            else:
                stacks.append(nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, 7, padding=3, groups=out_dim),
                    norm,
                    nn.Conv2d(out_dim, out_dim*expansion, 1, padding=0),
                    nn.GELU(),
                    nn.Conv2d(out_dim*expansion, out_dim, 1, padding=0)
                ))
            #for layer in stacks[-1]:
            #    if hasattr(layer, 'weight'):
            #        component.nn_init(layer, options.initializer)
        self.stacks = nn.ModuleList(stacks)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        
    def output_size(self):
        return self.size

    def forward(self, input, context):
        for stack in self.stacks:
            input = input + stack(input)
        return input
