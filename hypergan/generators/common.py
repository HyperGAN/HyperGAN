import tensorflow as tf
import numpy as np
import hyperchamber as hc

def repeating_block(component, net, output_channels, filter=None):
    config = component.config
    ops = component.ops
    if output_channels == 3:
        return standard_block(component, net, output_channels, filter=filter)
    for i in range(config.block_repeat_count):
        net = standard_block(component, net, output_channels, filter=filter)
        print("[generator] repeating block ", net)
    return net

def standard_block(component, net, output_channels, filter=None):
    config = component.config
    ops = component.ops
    net = ops.conv2d(net, filter, filter, 1, 1, output_channels)
    return net

def inception_block(component, net, output_channels, filter=None):
    config = component.config
    ops = component.ops
    activation = ops.lookup(config.activation)
    size = int(net.get_shape()[-1])
    batch_size = int(net.get_shape()[0])

    if output_channels == 3:
        return standard_block(component, net, output_channels)

    net1 = ops.conv2d(net, filter, filter, 1, 1, output_channels//3)
    net2 = ops.conv2d(net1, filter, filter, 1, 1, output_channels//3)
    net3 = ops.conv2d(net2, filter, filter, 1, 1, output_channels//3)
    net = tf.concat(axis=3, values=[net1, net2, net3])
    return net

def dense_block(component, net, output_channels, filter=None):
    config = component.config
    ops = component.ops
    if output_channels == 3:
        return standard_block(component, net, output_channels)
    net1 = standard_block(component, net, output_channels)
    net2 = standard_block(component, net, output_channels)
    net = tf.concat(axis=3, values=[net1, net2])
    return net
    


