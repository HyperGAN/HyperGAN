import torch.nn as nn
from hypergan.layer_shape import LayerShape
import hypergan as hg
import torch

class ChannelAttention(hg.Layer):
    """ Self attention Layer on channels from https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py """
    def __init__(self, component, args, options):
        super(ChannelAttention, self).__init__(component, args, options)
        self.channels = args[0]
        self.dims = list(component.current_size.dims).copy()
        in_dim = self.dims[0]
        padding = (0,0)
        if options.padding is not None:
            padding = (options.padding, options.padding)

        self.f = nn.Conv2d(in_channels = in_dim , out_channels = in_dim, kernel_size=1) 
        self.g = nn.Conv2d(in_channels = in_dim , out_channels = in_dim, kernel_size=1)
        self.h = nn.Conv2d(in_channels = in_dim , out_channels = in_dim, kernel_size=1)
        self.v = nn.Conv2d(in_channels = in_dim , out_channels = self.channels , kernel_size= options.filter or 1, padding=padding)

        component.nn_init(self.f, options.initializer)
        component.nn_init(self.g, options.initializer)
        component.nn_init(self.h, options.initializer)
        component.nn_init(self.v, options.initializer)
        self.softmax  = nn.Softmax(dim=1) #

    def output_size(self):
        return LayerShape(self.channels, self.dims[1], self.dims[2])

    def forward(self, input, context):
        x = input
        m_batchsize,C,width ,height = x.size()
        f  = self.f(x).view(m_batchsize,C,width*height)
        g =  self.g(x).view(m_batchsize,C,width*height).transpose(1,2)
        fg =  torch.bmm(f,g)
        attention_map = self.softmax(fg)
        h = self.h(x).view(m_batchsize,C,width*height).transpose(1,2)

        fgh = torch.bmm(h, attention_map )
        return self.v(fgh.transpose(1,2).view(x.shape))
