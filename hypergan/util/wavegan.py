import tensorflow as tf
import hypergan.vendor.wavenet
from hypergan.util.ops import *


def discriminator(config, x):
    batch_size = config['batch_size']*2
    activation = config['d_activation']
    dilations = config['g_mp3_dilations'];
    dilation_channels = config['g_mp3_dilation_channels']
    residual_channels = config['g_mp3_residual_channels']
    output_height = 256
    filter = config['g_mp3_filter']
    results = []
    result = x
    result=tf.reshape(result, [batch_size, 1, config['mp3_size'], config['channels']])
    result = dilation_layer(result,output_height,  0,1,int(result.get_shape()[3]), residual_channels, filter)
    #result = conv2d(result, residual_channels, name='d_start', k_w=3, k_h=1, d_h=1, d_w=2)
    #result = batch_norm(batch_size, name='d_bn1')(result)
    #result = activation(result)
    #result = conv2d(result, residual_channels, name='d_start2', k_w=3, k_h=1, d_h=1, d_w=1)
    #?result = activation(result)
    for index, dilation in enumerate(dilations):
        results.append(dilation_layer(result, output_height, index+1,dilation, residual_channels, dilation_channels, filter))
    #result = tf.concat(3, results)
    result = tf.add_n(results)

    layers=4
    depth=2
    k=residual_channels*4
    #TODO:bn
    for i in range(layers):
        if i != layers-1:
            result = dense_block_1d(result, k, activation, batch_size, 'transition', 'd_layers_transition_'+str(i))
        for j in range(depth):
            result = dense_block_1d(result, k, activation, batch_size, 'layer', 'd_layers_'+str(i)+"_"+str(j))


 
    filter_size_w = int(result.get_shape()[1])
    filter_size_h = int(result.get_shape()[2])
    filter = [1,filter_size_w,filter_size_h,1]
    stride = [1,filter_size_w,filter_size_h,1]
    result = tf.nn.avg_pool(result, ksize=filter, strides=stride, padding='SAME')
    result = tf.reshape(result,[batch_size,-1])
    print("end of d is", result)
    return result

def dilation_layer(result, output_height, index, dilation, residual_channels, dilation_channels, filter):
    batch_size = int(result.get_shape()[0])
    wavenet = shared.vendor.wavenet.WaveNet()
    size = int(result.get_shape()[2])


    weights_filter = tf.get_variable('d_dilated_filter_'+str(index), [1,filter, residual_channels, dilation_channels], initializer=tf.truncated_normal_initializer(stddev=0.2))
    weights_gate = tf.get_variable('d_dilated_gate_'+str(index), [1,filter, residual_channels, dilation_channels], initializer=tf.truncated_normal_initializer(stddev=0.2))

    conv_filter = wavenet._causal_dilated_conv(result, weights_filter, dilation)
    conv_filter = batch_norm(batch_size, name='d_lbn'+str(index))(conv_filter)
    #conv_filter = tf.nn.relu(conv_filter)
    #result = conv_filter
    conv_gate = wavenet._causal_dilated_conv(result, weights_gate, dilation)
    result = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)
    #height=30
    #result = tf.reshape(result, [batch_size,1,height,dilation_channels ])

    result = conv2d(result, result.get_shape()[3], name='d_dilated_out_'+str(index), k_w=1, k_h=1, d_h=1, d_w=1)
    return result

def generator(config, inputs, reuse=False):
    activation=config['g_activation']
    batch_size=config['batch_size']

    with(tf.variable_scope("generator", reuse=reuse)):
        original_z = tf.concat(1, inputs)
        prime=1
        proj_size=prime*125
        dims = 128
        result = linear(original_z, proj_size*dims, scope="g_lin_proj")
        result = tf.reshape(result, [config['batch_size'], 1,proj_size, dims])

        widenings = 3
        stride = 4
        zs = [None]
        size=int(result.get_shape()[3])
        sc_layers = config['g_skip_connections_layers']
        for i in range(widenings):
            result = tf.concat(3, [result, tf.random_uniform([batch_size, int(result.get_shape()[1]), int(result.get_shape()[2]), 2], -1, 1)])
            if(i==widenings-1):
                #result = block_deconv_1d(result, activation, batch_size, 'deconv', 'g_layers_a'+str(i), output_channels=config['channels']*2, stride=stride)
                result = residual_block_deconv_1d(result, activation, batch_size, 'deconv', 'g_layers_a'+str(i), output_channels=(config['channels']+5)*2, stride=stride)
                result = residual_block_deconv_1d(result, activation, batch_size, 'bottleneck', 'g_layers_bottleneck_'+str(i), channels=config['channels']*2)
            else:
                result = residual_block_deconv_1d(result, activation, batch_size, 'deconv', 'g_layers_'+str(i), stride=stride)
                result = residual_block_deconv_1d(result, activation, batch_size, 'identity', 'g_layers_i_'+str(i))
            print("g result", result)

        #result = batch_norm(batch_size, name='g_bn')(result)
        ##if(config['g_last_layer']):
        ##     result = config['g_last_layer'](result)
        #result = activation(result)
        #output_shape = [batch_size, 1, config['mp3_size'], config['channels']]

        #left = deconv2d(result, output_shape, name='g_end_convl', k_w=1, k_h=1, d_h=1, d_w=1)
        #result = left
        #left = batch_norm(batch_size, name='g_bn_left')(left)
        #right = deconv2d(result, output_shape, name='g_end_convr', k_w=1, k_h=1, d_h=1, d_w=1)
        #

        left = tf.slice(result,[0,0,0,0], [-1,-1,-1,config['channels']])
        right = tf.slice(result,[0,0,0,config['channels']], [-1,-1,-1,config['channels']])
        result = tf.nn.tanh(right)*tf.nn.sigmoid(left)
        #result = tf.reshape(result, output_shape)

        output_shape = [batch_size, config['mp3_size'], config['channels']]
        print("G OUTPUT", result)
        result = tf.reshape(result, output_shape)
        return result, None

def residual_block_deconv_1d(result, activation, batch_size,id,name, output_channels=None, stride=2, channels=None):
    size = int(result.get_shape()[-1])
    s = result.get_shape()
    print("S IS", s)
    if(id=='bottleneck'):
        output_shape = [s[0], s[1], s[2],channels]
        output_shape = [int(o) for o in output_shape]
        result = batch_norm(batch_size, name=name+'bn_pre')(result)
        result = activation(result)
        left = deconv2d(result, output_shape, name=name+'l', k_h=1, k_w=stride+1, d_h=1, d_w=1)
        left = batch_norm(batch_size, name=name+'bn')(left)
        left = activation(left)
        left = deconv2d(left, output_shape, name=name+'l2', k_h=1, k_w=stride+1, d_h=1, d_w=1)
        right = deconv2d(result, output_shape, name=name+'r', k_h=1, k_w=stride+1, d_h=1, d_w=1)
    elif(id=='identity'):
        output_shape = s
        output_shape = [int(o) for o in output_shape]
        left = result
        left = batch_norm(batch_size, name=name+'bn')(left)
        left = activation(left)
        left = deconv2d(left, output_shape, name=name+'l', k_h=1, k_w=stride+1, d_h=1, d_w=1)
        left = batch_norm(batch_size, name=name+'bn2')(left)
        left = activation(left)
        left = deconv2d(left, output_shape, name=name+'l2', k_h=1, k_w=stride+1, d_h=1, d_w=1)
        right = result
    elif(id=='deconv'):
        output_shape = [s[0], 1, s[2]*stride, s[3]//(stride//2)]
        if(output_channels):
            output_shape[-1] = output_channels
        output_shape = [int(o) for o in output_shape]
        result = batch_norm(batch_size, name=name+'bn')(result)
        result = activation(result)
        left = result
        right = result
        left = deconv2d(left, output_shape, name=name+'l', k_h=1, k_w=stride+1, d_h=1, d_w=stride)
        left = batch_norm(batch_size, name=name+'lbn')(left)
        left = activation(left)
        left = deconv2d(left, output_shape, name=name+'l2', k_h=1, k_w=stride+1, d_h=1, d_w=1)
        right = deconv2d(right, output_shape, name=name+'r', k_h=1, k_w=stride+1, d_h=1, d_w=stride)
    return left+right
def block_deconv_1d(result, activation, batch_size,id,name, output_channels=None, stride=2, channels=None):
    size = int(result.get_shape()[-1])
    s = result.get_shape()
    if(id=='deconv'):
        output_shape = [s[0], 1, s[2]*stride,s[3]//stride]
        if(output_channels):
            output_shape[-1] = output_channels
        output_shape = [int(o) for o in output_shape]
        result = batch_norm(batch_size, name=name+'bn')(result)
        result = activation(result)
        result = deconv2d(result, output_shape, name=name+'l', k_w=stride+1, k_h=1, d_h=1, d_w=stride)
    return result

def dense_block_1d(result, k, activation, batch_size, id, name):
    size = int(result.get_shape()[-1])
    if(id=='layer'):
        identity = tf.identity(result)
        result = batch_norm(batch_size, name=name+'bn')(result)
        result = activation(result)
        result = conv2d(result, k, name=name+'conv', k_w=3, k_h=1, d_h=1, d_w=1)

        return tf.concat(3,[identity, result])
    elif(id=='transition'):
        result = batch_norm(batch_size, name=name+'bn')(result)
        result = activation(result)
        result = conv2d(result, size, name=name+'id', k_w=1, k_h=1, d_h=1, d_w=1)
        filter = [1,1,4,1]
        stride = [1,1,4,1]
        result = tf.nn.avg_pool(result, ksize=filter, strides=stride, padding='SAME')
        return result


