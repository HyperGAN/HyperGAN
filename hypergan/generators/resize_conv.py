import tensorflow as tf
import numpy as np
from hypergan.util.hc_tf import *

def generator(config, net, z):
    depth=0
    w=int(net.get_shape()[1])
    target_w=int(config['x_dims'][0])
    h=int(net.get_shape()[2])
    target_h=int(config['x_dims'][1])
    while((w<target_w or w == 1) and h<target_h):
        if h != 1:
            h *= 2
        if w != 1:
            w *= 2
        depth += 1

    target_size = int(net.get_shape()[1])*(2**depth)*int(net.get_shape()[2])*(2**depth)*config['channels']
    nets=[]
    activation = config['generator.activation']
    batch_size = config['batch_size']
    depth_reduction = np.float32(config['generator.resize_conv.depth_reduction'])
    batch_norm = config['generator.regularizers.layer']

    s = [int(x) for x in net.get_shape()]


    print("s is", s)
    net = block_conv(net, activation, batch_size, 'identity', 'g_layers_init', output_channels=int(net.get_shape()[3]), filter=3, sigmoid_gate=z)
    print(net)
    if(config['generator.layer_filter']):
        fltr = config['generator.layer_filter'](None, net)
        if(fltr is not None):
            net = tf.concat(3, [net, fltr]) # TODO: pass through gan object

    print("DEPTH", depth)
    for i in range(depth):
        s = [int(x) for x in net.get_shape()]
        layers = int(net.get_shape()[3])//depth_reduction
        if(i == depth-1):
            layers=config['channels']
        resized_wh=[s[1], s[2]]
        if resized_wh[0] != 1:
            resized_wh[0] = min(config['x_dims'][0], resized_wh[0]*2)
        if resized_wh[1] != 1:
            resized_wh[1] = min(config['x_dims'][1], resized_wh[1]*2)
        if config['generator.layer.noise']:
            noise = [s[0],resized_wh[0],resized_wh[1],2**(depth-i)]
        else:
            noise = None
        net = tf.image.resize_images(net, [resized_wh[0], resized_wh[1]], 1)
        if(config['generator.layer_filter']):
            fltr = config['generator.layer_filter'](None, net)
            if(fltr is not None):
                net = tf.concat(3, [net, fltr]) # TODO: pass through gan object
                set_tensor('xfiltered', fltr)
        fltr = 3
        if fltr > net.get_shape()[1]:
            fltr=int(net.get_shape()[1])
        if fltr > net.get_shape()[2]:
            fltr=int(net.get_shape()[2])
        net = block_conv(net, activation, batch_size, 'identity', 'g_layers_'+str(i), output_channels=layers, filter=fltr, batch_norm=batch_norm, noise_shape=noise)
        if(i == depth-1):
            first3 = net
        else:
            first3 = tf.slice(net, [0,0,0,0], [-1,-1,-1,3])
        if batch_norm:
            first3 = batch_norm(config['batch_size'], name='g_bn_first3_'+str(i))(first3)
        first3 = config['generator.final_activation'](first3)
        nets.append(first3)
        size = int(net.get_shape()[1])*int(net.get_shape()[2])*int(net.get_shape()[3])
        print("[generator] layer",net, size, target_size,"  with noise ",noise)

    return nets


