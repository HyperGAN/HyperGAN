import numpy as np
import hyperchamber as hc
from hypergan.generators.configurable_generator import ConfigurableGenerator

import torch.nn as nn

from .base_generator import BaseGenerator

class ResizableGenerator(ConfigurableGenerator):

    def required(self):
        return "final_depth".split()

    def depths(self, initial_width=4):
        gan = self.gan
        config = self.config
        final_depth = config.final_depth
        depths = []

        target_w = gan.width()

        w = initial_width
        i = 0

        depths.append(final_depth)
        while w < target_w:
            w*=2
            depths.append(final_depth * 2**i)
            i+=1
        depths = depths[1:]
        depths.reverse()
        return depths

    def create(self):
        gan = self.gan
        config = self.config

        primes = config.initial_dimensions or [4, 4]
        self.initial_dimensions = primes
        depths = self.depths(primes[0])
        self.initial_depth = np.minimum(depths[0], config.max_depth or 512)

        self.current_input_size = self.gan.latent.config.z
        self.linear = nn.Sequential(self.layer_linear(None, [primes[0]*primes[1]*self.initial_depth], {}), nn.LeakyReLU())

        depths = self.depths(initial_width = self.initial_dimensions[0])

        depth_reduction = np.float32(config.depth_reduction)

        filter_size = config.filter or 3
        block = config.block

        self.current_height = self.initial_dimensions[1]
        self.current_width = self.initial_dimensions[0]
        self.current_channels = self.initial_depth
        conv_layers = []

        for i, depth in enumerate(depths[1:]):
            dep = np.minimum(depth, config.max_depth or 512)
            options = {}
            if block == 'deconv':
                options['stride'] = 2
                net = self.layer_deconv(None, [dep], options)
            elif block == 'subpixel':
                net = self.layer_subpixel(None, [dep], options)
            elif block == 'resize_conv':
                net = self.layer_resize_conv(None, [dep], options)
            conv_layers.append(net)
            conv_layers.append(nn.LeakyReLU())

        dep = config.channels or gan.channels()

        options = {}

        if block == 'deconv':
            options["stride"] = 2
            net = self.layer_deconv(None, [dep], options)

        elif block == "subpixel":
            net = self.layer_subpixel(None, [dep], options)

        elif block == "resize_conv":
            net = self.layer_resize_conv(None, [dep], options)

        conv_layers.append(net)
        conv_layers.append(nn.Tanh())
        self.net = nn.Sequential(*conv_layers)

    def forward(self, x):
        lin = self.linear(x)
        lin = lin.view(self.gan.batch_size(), self.initial_depth, self.initial_dimensions[0], self.initial_dimensions[1])
        net = self.net(lin)
        return net.view(self.gan.batch_size(), self.gan.channels(), self.gan.height(), self.gan.width())
