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

def multi_block(component, net, output_channels, filter=None):
    config = component.config
    ops = component.ops
    if output_channels == 3:
        return standard_block(component, net, output_channels, filter=filter)
    net = standard_block(component, net, output_channels, filter=filter)
    net2 = standard_block(component, net, output_channels, filter=filter, activation_regularizer=True)
    net3 = standard_block(component, net2, output_channels, filter=filter, activation_regularizer=True)
    return net+net2+net3


def standard_block(component, net, output_channels, filter=None, activation_regularizer=False, padding="SAME"):
    config = component.config
    ops = component.ops
    layer_regularizer = config.layer_regularizer

    if activation_regularizer:
        net = config.activation(net)
        if layer_regularizer is not None:
            net = component.layer_regularizer(net)

    if padding == "VALID":
        resize = [ops.shape(net)[1]+2, ops.shape(net)[2]+2]
        net = ops.resize_images(net, resize, config.resize_image_type or 1)
    net = ops.conv2d(net, filter, filter, 1, 1, output_channels, padding=padding)
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
    


