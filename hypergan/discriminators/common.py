import tensorflow as tf
import hyperchamber as hc

def repeating_block(component, net, depth, filter=3):
    ops = component.ops
    config = component.config
    layer_regularizer = config.layer_regularizer
    ksize = [1,filter-1,filter-1,1]
    stride = [1,filter-1,filter-1,1]
    for i in range(config.block_repeat_count-1):
        net = config.activation(net)
        if layer_regularizer is not None:
            net = component.layer_regularizer(net)
        net = ops.conv2d(net, 3, 3, 1, 1, depth)
        print("[discriminator] hidden layer", net)

    net = tf.nn.avg_pool(net, ksize=ksize, strides=stride, padding='SAME')
    print('[discriminator] layer', net)
    return net

def repeating_strided_block(component, net, depth, filter=3):
    ops = component.ops
    config = component.config
    layer_regularizer = config.layer_regularizer
    ksize = [1,filter-1,filter-1,1]
    stride = [1,filter-1,filter-1,1]
    for i in range(config.block_repeat_count-1):
        net = config.activation(net)
        if layer_regularizer is not None:
            net = component.layer_regularizer(net)
        if i== config.block_repeat_count-2:
            net = ops.conv2d(net, 3, 3, 2, 2, depth)
        else:
            net = ops.conv2d(net, 3, 3, 1, 1, depth)
        print("[discriminator] hidden layer", net)

    print('[discriminator] layer', net)
    return net


def standard_block(component, net, depth, filter=3):
    ops = component.ops
    config = component.config
    stride_w = filter-1
    stride_h = filter-1
    ksize = [1,filter-1,filter-1,1]
    stride = [1,stride_w,stride_h,1]

    net = ops.conv2d(net, filter, filter, 1, 1, depth)
    net = tf.nn.avg_pool(net, ksize=ksize, strides=stride, padding='SAME')
    print('[discriminator] layer', net)
    return net

def strided_block(component, net, depth, filter=3):
    ops = component.ops
    config = component.config
    net = ops.conv2d(net, filter, filter, 2, 2, depth)
    print('[discriminator] layer', net)
    return net
