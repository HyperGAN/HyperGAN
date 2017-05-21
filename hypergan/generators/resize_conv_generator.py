import tensorflow as tf
import numpy as np
import inspect
import hyperchamber as hc
from hypergan.generators.common import *

class ResizeConvGenerator:
    def __init__(self,
            z_projection_depth=512,
            activation='lrelu',
            final_activation='tanh',
            depth_reduction=2,
            layer_filter=None,
            layer_regularizer='batch_norm',
            block=[standard_block],
            resize_image_type=1,
            block_repeat_count=[2],
            batch_norm_momentum=[0.001],
            batch_norm_epsilon=[0.0001],
            orthogonal_initializer_gain=1

            ):
        selector = hc.Selector()

        selector.set("z_projection_depth", z_projection_depth) # Used in the first layer - the linear projection of z
        selector.set("activation", activation); # activation function used inside the generator
        selector.set("final_activation", final_activation) # Last layer of G.  Should match the range of your input - typically -1 to 1
        selector.set("depth_reduction", depth_reduction) # Divides our depth by this amount every time we go up in size
        selector.set('layer_filter', layer_filter) #Add information to g
        selector.set('layer_regularizer', layer_regularizer)
        selector.set('block', block)
        selector.set('block_repeat_count', block_repeat_count)
        selector.set('resize_image_type', resize_image_type)

        selector.set('orthogonal_initializer_gain', orthogonal_initializer_gain)
        selector.set('batch_norm_momentum', batch_norm_momentum)
        selector.set('batch_norm_epsilon', batch_norm_epsilon)
        self.config = selector.random_config()

    def create(self, gan, net):
        depth = 0
        primes = [4, 4]
        nets = []
        config = self.config
        x_dims = gan.config.x_dims
        gconfig = {k[2:]: v for k, v in gan.config.items() if k[2:] in inspect.getargspec(gan.ops).args}
        ops = gan.ops(*dict(gconfig))
        batch_size = gan.config.batch_size
        z_proj_dims = config.z_projection_depth
        new_shape = [batch_size, primes[0], primes[1], z_proj_dims]

        activation = ops.lookup(config.activation)
        final_activation = ops.lookup(config.final_activation)

        print('net', net)
        net = ops.linear(net, z_proj_dims*primes[0]*primes[1])
        net = ops.reshape(net, new_shape)
        print('net', net)

        w = ops.shape(net)[1]
        target_w = int(ops.shape(gan.graph.x)[0])

        while w < target_w:
            w*=2
            depth += 1

        depth_reduction = np.float32(config.depth_reduction)

        shape = ops.shape(net)

        net = config.block(ops, net, config, shape[3])
        net = self.layer_filter(gan, config, net)

        for i in range(depth):
            s = ops.shape(net)
            is_last_layer = (i == depth-1)

            reduced_layers = shape[3]-depth_reduction
            layers = gan.config.channels if is_last_layer else reduced_layers
            resize = [min(s[1]*2, x_dims[0]), min(s[2]*2, x_dims[1])]

            net = ops.resize_images(net, resize, config.resize_image_type)
            net = self.layer_filter(gan, config, net)
            net = config.block(ops, net, config, layers)

            sliced = ops.slice(net, [0,0,0,0], [-1,-1,-1, gan.config.channels])
            first3 = net if is_last_layer else sliced

            first3 = ops.layer_regularizer(first3, config.layer_regularizer, config.batch_norm_epsilon)

            first3 = final_activation(first3)

            nets.append(first3)
            size = resize[0]*resize[1]*layers
            print("[generator] layer", net, size)

        return nets

    def layer_filter(self, gan, config, net):
        if config.layer_filter:
            fltr = config.layer_filter(gan, net)
            if fltr is not None:
                net = ops.concat(axis=3, values=[net, fltr])
        return net
