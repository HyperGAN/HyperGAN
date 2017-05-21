import tensorflow as tf
import numpy as np
import hyperchamber as hc

def repeating_block(net, config, activation, batch_size,id,name, resize=None, output_channels=None, stride=2, noise_shape=None, dtype=tf.float32,filter=3, batch_norm=None, sigmoid_gate=None, reshaped_z_proj=None):
    if output_channels == 3:
        return standard_block(ops, net, config, output_channels)
    for i in range(config.block_repeat_count):
        net = standard_block(ops, net, config, output_channels)
        print("[generator] repeating block ", net)
    return net

def standard_block(ops, net, config, output_channels):
    activation = ops.lookup(config.activation)
    net = activation(net)
    net = ops.layer_regularizer(net, config.layer_regularizer, config.batch_norm_epsilon)
    net = ops.conv2d(net, 3, 3, 1, 1, output_channels)
    return net

def inception_block(ops, net, config, output_channels):
    activation = ops.lookup(config.activation)
    size = int(net.get_shape()[-1])

    if output_channels == 3:
        return block_conv(net, config, activation, batch_size, 'identity', name, output_channels=output_channels, filter=filter, batch_norm=config.layer_regularizer)

    net = ops.layer_regularizer(batch_size, momentum=config.batch_norm_momentum, epsilon=config.batch_norm_epsilon, name=name+'bn')(net)
    net = activation(net)

    net1 = ops.conv2d(net, output_channels//3, name=name+'1', k_w=1, k_h=1, d_h=1, d_w=1)
    net2 = ops.conv2d(net1, output_channels//3, name=name+'2', k_w=filter, k_h=filter, d_h=1, d_w=1)
    net3 = ops.conv2d(net2, output_channels//3, name=name+'3', k_w=filter, k_h=filter, d_h=1, d_w=1)
    net = tf.concat(axis=3, values=[net1, net2, net3])
    return net

def dense_block(ops, net, config, output_channels):
    if output_channels == 3:
        return standard_block(ops, net, config, output_channels)
    net1 = standard_block(ops, net, config, output_channels)
    net2 = standard_block(ops, net, config, output_channels)
    net = tf.concat(axis=3, values=[net1, net2])
    return net
    

generator_prelus=0
def generator_prelu(net):
    global generator_prelus # hack
    generator_prelus+=1
    return prelu('g_', generator_prelus, net) # Only ever 1 generator

def minmax(net):
    net = tf.minimum(net, 1)
    net = tf.maximum(net, -1)
    return net

def minmaxzero(net):
    net = tf.minimum(net, 1)
    net = tf.maximum(net, 0)
    return net
