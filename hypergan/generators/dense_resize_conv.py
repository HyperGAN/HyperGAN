import tensorflow as tf
import numpy as np
from hypergan.util.hc_tf import *

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
    batch_norm = config['generator.regularizers.layer']

    s = [int(x) for x in net.get_shape()]

    resize=[s[1]*2, s[2]*2]
    net = block_conv(net, activation, batch_size, 'identity', 'g_layers_init', output_channels=int(net.get_shape()[3]), filter=3, dropout=get_tensor('dropout'), resize = resize)

    z_proj = None
    layers = int(net.get_shape()[3])
    for i in range(depth):
        s = [int(x) for x in net.get_shape()]
        resized_wh=[s[1]*2, s[2]*2]
        layers = layers // depth_reduction
        if(i == depth-1):
            layers=config['channels']*2
        if config['generator.layer.noise']:
            noise = [s[0],resized_wh[0],resized_wh[1],2**(depth-i)]
        else:
            noise = None
        fltr = 3
        if fltr > net.get_shape()[1]:
            fltr=int(net.get_shape()[1])
        if fltr > net.get_shape()[2]:
            fltr=int(net.get_shape()[2])
        
        if i > 0:
            net = block_conv(net, activation, batch_size, 'identity', 'g_layers_'+str(i), output_channels=layers, filter=3, batch_norm=batch_norm, noise_shape=noise, resize=[resized_wh[0], resized_wh[1]])
        

        dense_size = config['generator.densenet.size']
        dense_depth = config['generator.densenet.layers']
        #dense_size = int(net.get_shape()[3])//4
        print("[generator] before dense layer",net, "  with noise ",noise)


        if i < depth - 1:
            for j in range(dense_depth):
                net2 = block_conv(net, activation, batch_size, 'identity', 'g_layers_dense_'+str(j)+"_"+str(i), output_channels=dense_size, filter=3, batch_norm=batch_norm, noise_shape=noise)
                nois = []
                s2 = [int(x) for x in net2.get_shape()]
                nv = tf.random_uniform([s2[0],s2[1],s2[2],s2[3]//4],-1.0, 1.0,dtype=config['dtype'])
                if j == 0:
                    nois.append(nv)
                net = tf.concat(3, [net, net2] + nois)
        #first3 = block_conv(net, activation, batch_size, 'identity', 'g_layers2_'+str(i), output_channels=config['channels'], filter=3, batch_norm=batch_norm, noise_shape=noise, resize=[resized_wh[0], resized_wh[1]])
        #last3 = block_conv(net, activation, batch_size, 'identity', 'g_layers3_'+str(i), output_channels=config['channels'], filter=3, batch_norm=batch_norm, noise_shape=noise, resize=[resized_wh[0], resized_wh[1]])

        first3 = tf.slice(net, [0,0,0,0], [-1,-1,-1,3])
        last3 = tf.slice(net, [0,0,0,3], [-1,-1,-1,3])
        if batch_norm:
            first3 = batch_norm(config['batch_size'], name='g_bn_first3_'+str(i))(first3)
            last3 = batch_norm(config['batch_size'], name='g_bn_last3'+str(i))(last3)
        first3 = config['generator.final_activation'](first3)
        last3 = tf.nn.sigmoid(last3)
        first3 = first3*last3
        nets.append(first3)
        size = int(net.get_shape()[1])*int(net.get_shape()[2])*int(net.get_shape()[3])
        print("[generator] layer",net, size, target_size,"  with noise ",noise)

    return nets


