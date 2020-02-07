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
        self.layer_linear(None, [primes[0]*primes[1]*self.initial_depth], {})
        self.linear = self.nn_layers[-1]
        #self.linear = self.layer_linear(None, [gan.width()*gan.height()*gan.channels()], {})

        depths = self.depths(initial_width = self.initial_dimensions[0])

        depth_reduction = np.float32(config.depth_reduction)

        filter_size = config.filter or 3
        block = config.block

        self.current_height = self.initial_dimensions[1]
        self.current_width = self.initial_dimensions[0]
        self.current_channels = self.initial_depth

        for i, depth in enumerate(depths[1:]):
            resize = [min(self.current_height, gan.height()), min(self.current_width, gan.width())]
            dep = np.minimum(depth, config.max_depth or 512)
            options = {"initializer": "he_normal", "avg_pool": 1, "stride": 1, "filter": 3}
            if block == 'deconv':
                options['stride'] = 2
                net = self.layer_deconv(None, [dep], options)
            elif block == 'subpixel':
                net = self.do_layer(self.layer_subpixel, net, [dep], options)
            elif block == 'resize_conv':
                net = self.do_layer(self.layer_resize_conv, net, [dep], options)

            size = resize[0]*resize[1]*depth

        dep = config.channels or gan.channels()

        options = {"avg_pool": 1, "stride": 1, "filter": 3, "activation": "null"}
        needs_resize = True

        if block == 'deconv':
            options["stride"] = 2
            net = self.layer_deconv(None, [dep], options)

        elif block == "subpixel":
            net = self.do_layer(self.layer_subpixel, net, [dep], options)

        elif block == "resize_conv":
            options["w"] = resize[0]
            options["h"] = resize[1]
            net = self.do_layer(self.layer_resize_conv, net, [dep], options)
            needs_resize = False

        self.nn_layers.append(nn.Tanh())
        self.net = nn.Sequential(*self.nn_layers)

    def forward(self, x):
        lin = self.linear(x)
        lin = lin.view(self.gan.batch_size(), self.initial_depth, self.initial_dimensions[0], self.initial_dimensions[1])
        net = self.net(lin)
        return net.view(self.gan.batch_size(), self.gan.channels(), self.gan.height(), self.gan.width())
