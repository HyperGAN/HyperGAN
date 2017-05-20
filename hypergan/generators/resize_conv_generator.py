import tensorflow as tf
import numpy as np
import inspect
import hyperchamber as hc
from hypergan.util.hc_tf import *
from hypergan.generators.common import *

class ResizeConvGenerator:

    def __init__(self,
            prefix = 'g_',
            z_projection_depth=512,
            activation=generator_prelu,
            final_activation=tf.nn.tanh,
            depth_reduction=2,
            layer_filter=None,
            layer_regularizer=batch_norm_1,
            block=[standard_block],
            resize_image_type=1,
            sigmoid_gate=False,
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
        selector.set('sigmoid_gate', sigmoid_gate)

        selector.set('orthogonal_initializer_gain', orthogonal_initializer_gain)
        selector.set('batch_norm_momentum', batch_norm_momentum)
        selector.set('batch_norm_epsilon', batch_norm_epsilon)
        self.config = selector.random_config()
        self.prefix = prefix

    def create(self, gan, net):
        depth = 0
        primes = [4, 4]
        nets = []
        config = self.config
        print("gan", gan)
        x_dims = gan.config.x_dims
        print("X_DIMS", x_dims)
        gconfig = {k[2:]: v for k, v in gan.config.items() if k[2:] in inspect.getargspec(gan.ops).args}
        print("CONFIG", config)
        ops = gan.ops(*dict(gconfig))
        batch_size = gan.config.batch_size
        z_proj_dims = config.z_projection_depth
        new_shape = [batch_size, primes[0], primes[1], z_proj_dims]

        activation = ops.lookup(config.activation)
        final_activation = ops.lookup(config.final_activation)

        net = ops.linear(net, z_proj_dims*primes[0]*primes[1])
        net = ops.reshape(net, new_shape)

        w = ops.shape(net)[1]
        target_w = int(ops.shape(gan.graph.x)[0])

        while w < target_w:
            w*=2
            depth += 1

        depth_reduction = np.float32(config.depth_reduction)

        s = ops.shape(net)

        net = config.block(net, config, output_channels=ops.shape(net)[3])
        net = self.layer_filter(gan, config, net)

        for i in range(depth):
            s = ops.shape(net)
            is_last_iteration = (i == depth-1)

            reduced_layers = s[3]-depth_reduction
            layers = gan.config.channels if is_last_iteration else reduced_layers
            resize = [min(s[1]*2, x_dims[0]), min(s[2]*2, x_dims[1])]

            net = ops.resize_images(net, resized_wh, config.resize_image_type)
            net = self.layer_filter(gan, config, net)
            net = config.block(net, config, output_channels=layers)

            sliced = ops.slice(net, [0,0,0,0], [-1,-1,-1, gan.config.channels])
            first3 = net if is_last_iteration else sliced

            first3 = ops.layer_regularizer(config.layer_regularizer, config.batch_norm_momentum, config.batch_norm_epsilon, first3)

            first3 = final_activation(first3)

            nets.append(first3)
            size = resize[1]*resize[2]*resize[3]
            print("[generator] layer", net, size)

        return nets

    def layer_filter(gan, config, net):
        if config.layer_filter:
            fltr = config.layer_filter(gan, net)
            if fltr is not None:
                net = ops.concat(axis=3, values=[net, fltr])
        return net

    def layer_regularizer(gan, config, net):
        if config.layer_regularizer:
            net = config.layer_regularizer(gan.config.batch_size, momentum=config.batch_norm_momentum, epsilon=config.batch_norm_epsilon, name=prefix+'bn_first3_'+str(i))(net)
        return net
