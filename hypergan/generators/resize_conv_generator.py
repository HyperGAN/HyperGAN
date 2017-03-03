import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.util.hc_tf import *

def standard_block(net, config, activation, batch_size,id,name, resize=None, output_channels=None, stride=2, noise_shape=None, dtype=tf.float32,filter=3, batch_norm=None, sigmoid_gate=None, reshaped_z_proj=None):
    return block_conv(net, activation, batch_size, 'identity', name, output_channels=output_channels, filter=filter, batch_norm=config.layer_regularizer)

def inception_block(net, config, activation, batch_size,id,name, resize=None, output_channels=None, stride=2, noise_shape=None, dtype=tf.float32,filter=3, batch_norm=None, sigmoid_gate=None, reshaped_z_proj=None):
    if output_channels == 3:
        return block_conv(net, activation, batch_size, 'identity', name, output_channels=output_channels, filter=filter, batch_norm=config.layer_regularizer)
    size = int(net.get_shape()[-1])
    if(batch_norm is not None):
        net = batch_norm(batch_size, name=name+'bn')(net)

    net = activation(net)
    s = net.get_shape()
    if(sigmoid_gate is not None):
        mask = linear(sigmoid_gate, s[1]*s[2]*s[3], scope=name+"lin_proj_mask")
        mask = tf.reshape(mask, net.get_shape())
        net *= tf.nn.sigmoid(mask)

    if output_channels == 3:
        return conv2d(net, output_channels, name=name, k_w=filter, k_h=filter, d_h=1, d_w=1)

    net1 = conv2d(net, output_channels//3, name=name+'1', k_w=1, k_h=1, d_h=1, d_w=1)
    net2 = conv2d(net1, output_channels//3, name=name+'2', k_w=filter, k_h=filter, d_h=1, d_w=1)
    net3 = conv2d(net2, output_channels//3, name=name+'3', k_w=filter, k_h=filter, d_h=1, d_w=1)
    net = tf.concat(axis=3, values=[net1, net2, net3])
    return net

def dense_block(net,config,  activation, batch_size,id,name, resize=None, output_channels=None, stride=2, noise_shape=None, dtype=tf.float32,filter=3, batch_norm=None, sigmoid_gate=None, reshaped_z_proj=None):
    if output_channels == 3:
        return block_conv(net, activation, batch_size, 'identity', name, output_channels=output_channels, filter=filter, batch_norm=config.layer_regularizer)

    net1 = block_conv(net, activation, batch_size, 'identity', name, output_channels=max(output_channels-16, 16), filter=filter, batch_norm=config.layer_regularizer)
    net2 = block_conv(net, activation, batch_size, 'identity', name+'2', output_channels=16, filter=filter, batch_norm=config.layer_regularizer)
    net = tf.concat(axis=3, values=[net1, net2])
    return net

generator_prelus=0
def generator_prelu(net):
    global generator_prelus # hack
    generator_prelus+=1
    return prelu('g_', generator_prelus, net) # Only ever 1 generator

def config(
        z_projection_depth=512,
        activation=generator_prelu,
        final_activation=tf.nn.tanh,
        depth_reduction=2,
        layer_filter=None,
        layer_regularizer=batch_norm_1,
        block=[standard_block],
        resize_image_type=1,
        sigmoid_gate=False,
        create_method=None
        ):
    selector = hc.Selector()
    
    if create_method is None:
       selector.set('create', create)
    else:
        selector.set('create', create_method)

    selector.set("z_projection_depth", z_projection_depth) # Used in the first layer - the linear projection of z
    selector.set("activation", activation); # activation function used inside the generator
    selector.set("final_activation", final_activation); # Last layer of G.  Should match the range of your input - typically -1 to 1
    selector.set("depth_reduction", depth_reduction) # Divides our depth by this amount every time we go up in size
    selector.set('layer_filter', layer_filter) #Add information to g
    selector.set('layer_regularizer', batch_norm_1)
    selector.set('block', block)
    selector.set('resize_image_type', resize_image_type)
    selector.set('sigmoid_gate', sigmoid_gate)

    return selector.random_config()

def create(config, gan, net):
    z = net
    x_dims = gan.config.x_dims
    z_proj_dims = config.z_projection_depth
    primes = find_smallest_prime(x_dims[0], x_dims[1])
    # project z
    net = linear(net, z_proj_dims*primes[0]*primes[1], scope="g_lin_proj")
    new_shape = [gan.config.batch_size, primes[0],primes[1],z_proj_dims]
    net = tf.reshape(net, new_shape)

    depth=0
    w=int(net.get_shape()[1])
    target_w=int(gan.config.x_dims[0])
    while(w<target_w):
        w*=2
        depth += 1

    nets=[]
    activation = config.activation
    batch_size = gan.config.batch_size
    depth_reduction = np.float32(config.depth_reduction)

    s = [int(x) for x in net.get_shape()]

    net = config.block(net, config, activation, batch_size, 'identity', 'g_layers_init', output_channels=int(net.get_shape()[3]), filter=3)
    if(config.layer_filter):
        fltr = config.layer_filter(gan, net)
        if(fltr is not None):
            net = tf.concat(axis=3, values=[net, fltr]) # TODO: pass through gan object

    for i in range(depth):
        s = [int(x) for x in net.get_shape()]
        layers = int(net.get_shape()[3])//depth_reduction
        if(i == depth-1):
            layers=gan.config.channels
        resized_wh=[s[1]*2, s[2]*2]
        net = tf.image.resize_images(net, [resized_wh[0], resized_wh[1]], config.resize_image_type)
        if(config.layer_filter):
            fltr = config.layer_filter(gan, net)
            if(fltr is not None):
                net = tf.concat(axis=3, values=[net, fltr]) # TODO: pass through gan object
        fltr = 3
        if fltr > net.get_shape()[1]:
            fltr=int(net.get_shape()[1])
        if fltr > net.get_shape()[2]:
            fltr=int(net.get_shape()[2])

        if config.sigmoid_gate:
            sigmoid_gate = z
        else:
            sigmoid_gate = None

        net = config.block(net, config, activation, batch_size, 'identity', 'g_layers_'+str(i), output_channels=layers, filter=3, sigmoid_gate=sigmoid_gate)
        if(i == depth-1):
            first3 = net
        else:
            first3 = tf.slice(net, [0,0,0,0], [-1,-1,-1,3])
        if config.final_activation:
            if config.layer_regularizer:
                first3 = config.layer_regularizer(gan.config.batch_size, name='g_bn_first3_'+str(i))(first3)
            first3 = config.final_activation(first3)
        nets.append(first3)
        size = int(net.get_shape()[1])*int(net.get_shape()[2])*int(net.get_shape()[3])
        print("[generator] layer", net, size)

    return nets

    
def minmax(net):
    net = tf.minimum(net, 1)
    net = tf.maximum(net, -1)
    return net
