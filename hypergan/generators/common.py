import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.util.hc_tf import *

def repeating_block(net, config, activation, batch_size,id,name, resize=None, output_channels=None, stride=2, noise_shape=None, dtype=tf.float32,filter=3, batch_norm=None, sigmoid_gate=None, reshaped_z_proj=None):
    if output_channels == 3:
        return block_conv(net, config, activation, batch_size, 'identity', name, output_channels=output_channels, filter=filter, batch_norm=config.layer_regularizer)
    for i in range(config.block_repeat_count):
        net = block_conv(net, config, activation, batch_size, 'identity', name+"_"+str(i), output_channels=output_channels, filter=filter, batch_norm=config.layer_regularizer)
        print("[generator] repeating block ", net)
    return net

def standard_block(net, config, activation, batch_size,id,name, resize=None, output_channels=None, noise_shape=None, dtype=tf.float32,filter=3, batch_norm=None, sigmoid_gate=None, reshaped_z_proj=None):
    return block_conv(net, config, activation, batch_size, 'identity', name, output_channels=output_channels, filter=filter, batch_norm=config.layer_regularizer)

def inception_block(net, config, activation, batch_size,id,name, resize=None, output_channels=None, noise_shape=None, dtype=tf.float32,filter=3, batch_norm=None, sigmoid_gate=None, reshaped_z_proj=None):
    if output_channels == 3:
        return block_conv(net, config, activation, batch_size, 'identity', name, output_channels=output_channels, filter=filter, batch_norm=config.layer_regularizer)
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

def dense_block(net,config,  activation, batch_size,id,name, resize=None, output_channels=None, noise_shape=None, dtype=tf.float32,filter=3, batch_norm=None, sigmoid_gate=None, reshaped_z_proj=None):
    if output_channels == 3:
        return block_conv(net, config, activation, batch_size, 'identity', name, output_channels=output_channels, filter=filter, batch_norm=config.layer_regularizer)

    net1 = block_conv(net, config, activation, batch_size, 'identity', name, output_channels=max(output_channels-16, 16), filter=filter, batch_norm=config.layer_regularizer)
    net2 = block_conv(net, config, activation, batch_size, 'identity', name+'2', output_channels=16, filter=filter, batch_norm=config.layer_regularizer)
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
