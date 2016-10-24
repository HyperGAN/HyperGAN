import tensorflow as tf
import numpy as np
from lib.util.hc_tf import *

def generator(config, net):
    depth=0
    w=int(net.get_shape()[1])
    target_w=int(config['x_dims'][0])
    while(w<target_w):
      w*=2
      depth +=1

    target_size = int(net.get_shape()[1])*(2**depth)*int(net.get_shape()[2])*(2**depth)*config['channels']
    nets=[]
    activation = config['generator.activation']
    batch_size = config['batch_size']
    depth_reduction = np.float32(config['generator.resize_conv.depth_reduction'])

    for i in range(depth):
        s = [int(x) for x in net.get_shape()]
        layers = int(net.get_shape()[3])//depth_reduction
        if(i == depth-1):
            layers=config['channels']
        resized_wh=[s[1]*2, s[2]*2]
        net = tf.image.resize_images(net, resized_wh[0], resized_wh[1], 1)
        noise = [s[0],resized_wh[0],resized_wh[1],2**(depth+1-i)]
        net = block_conv(net, activation, batch_size, 'identity', 'g_layers_'+str(i), output_channels=layers, filter=3, noise_shape=noise)
        first3 = tf.slice(net, [0,0,0,0], [-1,-1,-1,3])
        first3 = batch_norm(config['batch_size'], name='g_bn_first3_'+str(i))(first3)
        first3 = config['generator.final_activation'](first3)
        nets.append(first3)
        size = int(net.get_shape()[1])*int(net.get_shape()[2])*int(net.get_shape()[3])
        print("Generator layer:",net, size, target_size)

    return nets


