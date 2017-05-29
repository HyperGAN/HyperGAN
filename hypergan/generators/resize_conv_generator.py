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
        create_method=None,
	block_repeat_count=[2],
	batch_norm_momentum=[0.001],
	batch_norm_epsilon=[0.0001],
	orthogonal_initializer_gain=1

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
    selector.set('layer_regularizer', layer_regularizer)
    selector.set('block', block)
    selector.set('block_repeat_count', block_repeat_count)
    selector.set('resize_image_type', resize_image_type)
    selector.set('sigmoid_gate', sigmoid_gate)

    selector.set('orthogonal_initializer_gain', orthogonal_initializer_gain)
    selector.set('batch_norm_momentum', batch_norm_momentum)
    selector.set('batch_norm_epsilon', batch_norm_epsilon)
    return selector.random_config()

def create(config, gan, net, prefix="g_"):
    print("CREATING g from", net)
    z = net
    x_dims = gan.config.x_dims
    z_proj_dims = config.z_projection_depth
    primes = find_smallest_prime(x_dims[0], x_dims[1])
    print("PRIMES", primes)
    # project z
    net = linear(net, z_proj_dims*primes[0]*primes[1], scope=prefix+"lin_proj", gain=config.orthogonal_initializer_gain)
    new_shape = [gan.config.batch_size, primes[0],primes[1],z_proj_dims]
    net = tf.reshape(net, new_shape)

    print('---', net)

    depth=0
    w=int(net.get_shape()[1])
    target_w=int(gan.config.x_dims[0])
    while(w<target_w):
        w*=2
        depth += 1

    nets=[]
    activation = config.activation
    batch_size = gan.config.batch_size
    depth_reduction = np.float32(config.depth_reduction)

    s = [int(x) for x in net.get_shape()]

    net = config.block(net, config, activation, batch_size, 'identity', prefix+'layers_init', output_channels=int(net.get_shape()[3]), filter=3)
    if(config.layer_filter):
        fltr = config.layer_filter(gan, net)
        if(fltr is not None):
            net = tf.concat(axis=3, values=[net, fltr]) # TODO: pass through gan object

    for i in range(depth):
        s = [int(x) for x in net.get_shape()]
        
        #layers = int(net.get_shape()[3])//depth_reduction
        layers = int(net.get_shape()[3])-depth_reduction
        if(i == depth-1):
            layers=gan.config.channels
        resized_wh=[s[1]*2, s[2]*2]
        if(resized_wh[0] > x_dims[0]):
            resized_wh[0]=x_dims[0]
        if(resized_wh[1] > x_dims[1]):
            resized_wh[1]=x_dims[1]
        print(';;;;',resized_wh)
        if gan.config.x_dims[1] == 1:
            resized_wh[1]=1
        net = tf.image.resize_images(net, [resized_wh[0], resized_wh[1]], config.resize_image_type)

        print('---', net)
        if(config.layer_filter):
            fltr = config.layer_filter(gan, net)
            if(fltr is not None):
                net = tf.concat(axis=3, values=[net, fltr]) # TODO: pass through gan object
        fltr = 3
        if fltr > net.get_shape()[1]:
            fltr=int(net.get_shape()[1])
        if fltr > net.get_shape()[2]:
            fltr=int(net.get_shape()[2])

        if config.sigmoid_gate:
            sigmoid_gate = z
        else:
            sigmoid_gate = None

        if gan.config.x_dims[1] == 1:
            resized_wh[1]=1
        net = tf.image.resize_images(net, [resized_wh[0], resized_wh[1]], config.resize_image_type)

        print('---', net, layers)
        if(config.layer_filter):
            fltr = config.layer_filter(gan, net)
            if(fltr is not None):
                net = tf.concat(axis=3, values=[net, fltr]) # TODO: pass through gan object
        fltr = 3
        if fltr > net.get_shape()[1]:
            fltr=int(net.get_shape()[1])
        if fltr > net.get_shape()[2]:
            fltr=int(net.get_shape()[2])

        if config.sigmoid_gate:
            sigmoid_gate = z
        else:
            sigmoid_gate = None

        net = config.block(net, config, activation, batch_size, 'identity', prefix+'layers_'+str(i), output_channels=layers, filter=3, sigmoid_gate=sigmoid_gate)
        if(i == depth-1):
            first3 = net
        else:
            first3 = tf.slice(net, [0,0,0,0], [-1,-1,-1, gan.config.channels])
        if config.final_activation:
            if config.layer_regularizer:
                first3 = config.layer_regularizer(gan.config.batch_size, momentum=config.batch_norm_momentum, epsilon=config.batch_norm_epsilon, name=prefix+'bn_first3_'+str(i))(first3)
            first3 = config.final_activation(first3)
        nets.append(first3)
        size = int(net.get_shape()[1])*int(net.get_shape()[2])*int(net.get_shape()[3])
        print("[generator] layer", net, size)

    return nets

    
def minmax(net):
    net = tf.minimum(net, 1)
    net = tf.maximum(net, -1)
    return net

def minmaxzero(net):
    net = tf.minimum(net, 1)
    net = tf.maximum(net, 0)
    return net
