import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.util.hc_tf import *
from hypergan.generators.common import *

def config(
        z_projection_depth=512,
        activation=generator_prelu,
        final_activation=tf.nn.tanh,
        depth_reduction=2,
        layer_filter=None,
        layer_regularizer=batch_norm_1,
        block=[standard_block],
        resize_image_type=1,
        sigmoid_gate=False,
        create_method=None
        ):
    selector = hc.Selector()
    
    if create_method is None:
       selector.set('create', create)
    else:
        selector.set('create', create_method)

    selector.set("z_projection_depth", z_projection_depth) # Used in the first layer - the linear projection of z
    selector.set("activation", activation); # activation function used inside the generator
    selector.set("final_activation", final_activation); # Last layer of G.  Should match the range of your input - typically -1 to 1
    selector.set("depth_reduction", depth_reduction) # Divides our depth by this amount every time we go up in size
    selector.set('layer_filter', layer_filter) #Add information to g
    selector.set('layer_regularizer', batch_norm_1)
    selector.set('block', block)
    selector.set('resize_image_type', resize_image_type)
    selector.set('sigmoid_gate', sigmoid_gate)
    selector.set('extra_layers', 5)

    return selector.random_config()

def create(config, gan, net, input=None):
    z = net
    x_dims = gan.config.x_dims
    z_proj_dims = config.z_projection_depth
    primes = [x_dims[0],x_dims[1]]
    # project z
    print("Z_PRO", z_proj_dims, primes)
    net = linear(net, z_proj_dims*primes[0]*primes[1], scope="g_lin_proj", gain=config.orthogonal_initializer_gain)
    new_shape = [gan.config.batch_size, primes[0],primes[1],z_proj_dims]
    print("____", new_shape, net)
    net = tf.reshape(net, new_shape)

    original_z = net

    w=int(net.get_shape()[1])
    nets=[]
    activation = config.activation
    batch_size = gan.config.batch_size

    s = [int(x) for x in net.get_shape()]

    if(config.layer_filter):
        fltr = config.layer_filter(gan, net)
        if(fltr is not None):
            net = tf.concat(axis=3, values=[net, fltr]) # TODO: pass through gan object

    for i in range(config.depth):
        s = [int(x) for x in net.get_shape()]
        layers = int(net.get_shape()[3])
        #if(config.layer_filter):
        #    fltr = config.layer_filter(gan, net)
        #    if(fltr is not None):
        #        net = tf.concat(axis=3, values=[net, fltr]) # TODO: pass through gan object
        fltr = 3
        if fltr > net.get_shape()[1]:
            fltr=int(net.get_shape()[1])
        if fltr > net.get_shape()[2]:
            fltr=int(net.get_shape()[2])

        if config.sigmoid_gate:
            sigmoid_gate = z
        else:
            sigmoid_gate = None

        noise = tf.random_normal(net.get_shape(), mean=0, stddev=0.001)
        #net = tf.concat([net,noise], 3)
        print("INPUT IS", input)
        #net = tf.concat([net,input], 3)
        #net = tf.concat([net,original_z], 3)
        name= 'g_layers_end'
        output_channels = (gan.config.channels+(i+1))
        #net = tf.reshape(net, [gan.config.batch_size, primes[0], primes[1], -1])
        if i == config.depth-1:
            output_channels = gan.config.channels
        net = config.block(net, config, activation, batch_size, 'identity', 'g_laendyers_'+str(i), output_channels=output_channels, filter=3, sigmoid_gate=sigmoid_gate)
        #net = tf.reshape(net, [gan.config.batch_size, primes[0]*4, primes[1]*4, -1])
        first3 = net
        if config.final_activation:
            if config.layer_regularizer:
                first3 = config.layer_regularizer(gan.config.batch_size, name='g_bn_first3_'+str(i))(first3)
            first3 = config.final_activation(first3)
        nets.append(first3)
        size = int(net.get_shape()[1])*int(net.get_shape()[2])*int(net.get_shape()[3])
        print("[generator] layer", net, size)

    return nets

    
def minmax(net):
    net = tf.minimum(net, 1)
    net = tf.maximum(net, -1)
    return net
