import tensorflow as tf
import numpy as np
import hyperchamber as hc

def repeating_block(ops, net, config, output_channels):
    if output_channels == 3:
        return standard_block(ops, net, config, output_channels)
    for i in range(config.block_repeat_count):
        net = standard_block(ops, net, config, output_channels)
        print("[generator] repeating block ", net)
    return net

def standard_block(ops, net, config, output_channels):
    print("CONFIG ACT", config.activation)
    activation = ops.lookup(config.activation)
    print("nACT", activation)
    net = activation(net)
    net = ops.layer_regularizer(net, config.layer_regularizer, config.batch_norm_epsilon)
    net = ops.conv2d(net, 3, 3, 1, 1, output_channels)
    return net

def inception_block(ops, net, config, output_channels):
    activation = ops.lookup(config.activation)
    size = int(net.get_shape()[-1])
    batch_size = int(net.get_shape()[0])

    if output_channels == 3:
        return standard_block(ops, net, config, output_channels)

    net = ops.layer_regularizer(net, config.layer_regularizer, config.batch_norm_epsilon)
    net = activation(net)

    net1 = ops.conv2d(net, 3, 3, 1, 1, output_channels//3)
    net2 = ops.conv2d(net1, 3, 3, 1, 1, output_channels//3)
    net3 = ops.conv2d(net2, 3, 3, 1, 1, output_channels//3)
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
