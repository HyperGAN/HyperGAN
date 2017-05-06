import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.util.hc_tf import *
from hypergan.generators.common import *
import hypergan

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


def create(config, gan, net, prefix="g_"):
    # 1 create xab(xa), xba(xb) v
    # 2 add xa, xb to g : remove z, encode(x) as z v
    # 3 add distance(xba(xab(xa)), xa)
    # 4 add distance(xab(xba(xb)), xb)
    # 4 add distance(xab(xa), xb)
    # 5 add distance(xba(xb), xa))

    z_encoded = gan.graph.z_encoded

    # TODO Chain together gab(gba) as gabba
    if 'shared-pyramid' in config:
        z_encoded = gan.graph.z_encoded
        z_encoded2 = gan.graph.z_encoded
        if 'unique_z' in config:
            z_encoded2 = gan.graph.z_encoded2

        gan.graph.xab = create_g_pyramid(config, gan, gan.graph.xa, id=1, prefix="g_")
        gan.graph.xba = create_g_pyramid(config, gan, gan.graph.xb, id=0, prefix="g_", reuse=True)

        if 'separate_g' in config:
            gan.graph.ga = create_g_pyramid_from_z(config, gan, z_encoded, prefix="g_a")
            gan.graph.gb = create_g_pyramid_from_z(config, gan, z_encoded2, prefix="g_b")
        else:
            #gan.graph.gb = create_g_pyramid(config, gan, tf.zeros_like(gan.graph.xa), z=z_encoded, id=1, prefix="g_", reuse=True)
            #gan.graph.ga = create_g_pyramid(config, gan, tf.zeros_like(gan.graph.xb), z=z_encoded2, id=0, prefix="g_", reuse=True)
            gan.graph.ga = create_g_pyramid_from_z(config, gan, z_encoded, id=0, prefix="g_", reuse=True)
            gan.graph.gb = create_g_pyramid_from_z(config, gan, z_encoded2, id=1, prefix="g_", reuse=True)

        gan.graph.gab = create_g_pyramid(config, gan, gan.graph.ga,prefix="g_", id=1, reuse=True)
        gan.graph.gba = create_g_pyramid(config, gan, gan.graph.gb, prefix="g_", id=0, reuse=True)

        gan.graph.gabba = create_g_pyramid(config, gan, gan.graph.gab, prefix="g_", id=0, reuse=True)
        gan.graph.gbaab = create_g_pyramid(config, gan, gan.graph.gba, prefix="g_", id=1, reuse=True)

        gan.graph.xbaab = create_g_pyramid(config, gan, gan.graph.xba, prefix="g_", id=1, reuse=True)
        gan.graph.xabba = create_g_pyramid(config, gan, gan.graph.xab, prefix="g_", id=0, reuse=True)

    elif 'shared-pyramid2' in config:
        z_encoded = gan.graph.z_encoded
        z_encoded2 = gan.graph.z_encoded
        if 'unique_z' in config:
            z_encoded2 = gan.graph.z_encoded2

        gan.graph.xab = create_g_pyramid_2(config, gan, gan.graph.xa, id=1, prefix="g_")
        gan.graph.xba = create_g_pyramid_2(config, gan, gan.graph.x, id=0, prefix="g_", reuse=True)

        if 'separate_g' in config:
            gan.graph.ga = create_g_pyramid_from_z(config, gan, z_encoded, prefix="g_a")
            gan.graph.gb = create_g_pyramid_from_z(config, gan, z_encoded2, prefix="g_b")
        else:
            #gan.graph.gb = create_g_pyramid(config, gan, tf.zeros_like(gan.graph.xa), z=z_encoded, id=1, prefix="g_", reuse=True)
            #gan.graph.ga = create_g_pyramid(config, gan, tf.zeros_like(gan.graph.xb), z=z_encoded2, id=0, prefix="g_", reuse=True)
            gan.graph.ga = create_g_pyramid_from_z_2(config, gan, z_encoded, id=0, prefix="g_", reuse=True)
            gan.graph.gb = create_g_pyramid_from_z_2(config, gan, z_encoded2, id=1, prefix="g_", reuse=True)

        gan.graph.gab = create_g_pyramid_2(config, gan, gan.graph.ga,prefix="g_", id=1, reuse=True)
        gan.graph.gba = create_g_pyramid_2(config, gan, gan.graph.gb, prefix="g_", id=0, reuse=True)

        gan.graph.gabba = create_g_pyramid_2(config, gan, gan.graph.gab, prefix="g_", id=0, reuse=True)
        gan.graph.gbaab = create_g_pyramid_2(config, gan, gan.graph.gba, prefix="g_", id=1, reuse=True)

        gan.graph.xbaab = create_g_pyramid_2(config, gan, gan.graph.xba, prefix="g_", id=1, reuse=True)
        gan.graph.xabba = create_g_pyramid_2(config, gan, gan.graph.xab, prefix="g_", id=0, reuse=True)


    elif('pyramid' in config):

        z_encoded = gan.graph.z_encoded
        z_encoded2 = gan.graph.z_encoded
        if 'unique_z' in config:
            z_encoded2 = gan.graph.z_encoded2

        #gan.graph.xab = create_g_pyramid(config, gan, gan.graph.xa, z=tf.zeros_like(z_encoded), id=1, prefix="g_b")
        #gan.graph.xba = create_g_pyramid(config, gan, gan.graph.xb, z=tf.zeros_like(z_encoded), id=0, prefix="g_a")
        gan.graph.xab = create_g_pyramid(config, gan, gan.graph.xa, id=1, prefix="g_b")
        gan.graph.xba = create_g_pyramid(config, gan, gan.graph.xb, id=0, prefix="g_a")

        if 'separate_g' in config:
            gan.graph.ga = create_g_pyramid_from_z(config, gan, z_encoded, prefix="g_a")
            gan.graph.gb = create_g_pyramid_from_z(config, gan, z_encoded2, prefix="g_b")
        else:
            #gan.graph.gb = create_g_pyramid(config, gan, tf.zeros_like(gan.graph.xa), z=z_encoded, id=1, prefix="g_b", reuse=True)
            #gan.graph.ga = create_g_pyramid(config, gan, tf.zeros_like(gan.graph.xb), z=z_encoded2, id=0, prefix="g_a", reuse=True)
            gan.graph.ga = create_g_pyramid_from_z(config, gan, z_encoded, id=0, prefix="g_a", reuse=True)
            gan.graph.gb = create_g_pyramid_from_z(config, gan, z_encoded2, id=1, prefix="g_b", reuse=True)

        gan.graph.gab = create_g_pyramid(config, gan, gan.graph.ga, prefix="g_b", id=1, reuse=True)
        gan.graph.gba = create_g_pyramid(config, gan, gan.graph.gb, prefix="g_a", id=0, reuse=True)

        gan.graph.gabba = create_g_pyramid(config, gan, gan.graph.gab, prefix="g_a", id=0, reuse=True)
        gan.graph.gbaab = create_g_pyramid(config, gan, gan.graph.gba, prefix="g_b", id=1, reuse=True)

        gan.graph.xbaab = create_g_pyramid(config, gan, gan.graph.xba, prefix="g_b", id=1, reuse=True)
        gan.graph.xabba = create_g_pyramid(config, gan, gan.graph.xab, prefix="g_a", id=0, reuse=True)
    else:
        gan.graph.xab = create_g(config, gan, gan.graph.xa, prefix="g_ab_")[0]
        gan.graph.xba = create_g(config, gan, gan.graph.xb, prefix="g_ba_")[0]

        gan.graph.xabba = create_g(config, gan, gan.graph.xab, prefix="g_ba_", reuse=True)[0]
        gan.graph.xbaab = create_g(config, gan, gan.graph.xba, prefix="g_ab_", reuse=True)[0]
        
        za = create_g_pyramid_from_z(config, gan, z_encoded, prefix="g_za_", reuse=False)
        zb = create_g_pyramid_from_z(config, gan, z_encoded, prefix="g_zb_", reuse=False)

        gan.graph.gb = create_g(config, gan, za, prefix="g_ab_", reuse=True)[0]
        gan.graph.ga = create_g(config, gan, zb, prefix="g_ba_", reuse=True)[0]
        gan.graph.gab = create_g(config, gan, gan.graph.ga, prefix="g_ab_", reuse=True)[0]
        gan.graph.gba = create_g(config, gan, gan.graph.gb, prefix="g_ba_", reuse=True)[0]
        gan.graph.gbaab = create_g(config, gan, gan.graph.gba, prefix="g_ab_", reuse=True)[0]
        gan.graph.gabba = create_g(config, gan, gan.graph.gab, prefix="g_ba_", reuse=True)[0]

    return [gan.graph.gab, gan.graph.gba, gan.graph.gabba, gan.graph.gbaab]

def recode(config, gan, z, prefix="g_"):
    with tf.variable_scope("recode", reuse=False):
        net = linear(z, int(z.get_shape()[1]), scope=prefix+"lin_pro_z", gain=config.orthogonal_initializer_gain)
        net = config.final_activation(net)
        return net
def create_g_pyramid_from_z(config, gan, z, prefix="g_", id=0, reuse=False):
    with tf.variable_scope("autoencoder", reuse=reuse):
        gconfig = gan.config.generator_autoencode
        generator = hc.Config(hc.lookup_functions(gconfig))
        bs = int(z.get_shape()[0])
        const = tf.one_hot([id], 2)
        const = tf.reshape(const,[1, 2])
        const = tf.tile(const, [bs, 1])
        z = tf.concat([z, const], 1)
        print("z is ", z)
        rx = generator.create(generator, gan, z, prefix=prefix)[-1]
    
    return rx


def create_g_pyramid(config, gan, x, z=None, prefix="g_", id=0, reuse=False):
    with tf.variable_scope("autoencoder", reuse=reuse):
        dconfig = gan.config.discriminators[0]
        dconfig = hc.Config(hc.lookup_functions(dconfig))

        if('align_regularizer' in dconfig):
            dconfig['layer_regularizer'] = config['layer_regularizer']
        g = x
        net = hypergan.discriminators.pyramid_discriminator.discriminator(gan, dconfig, x, g, [x], [g], prefix)
        print("NET is ", net)
        s = [int(x) for x in net.get_shape()]
        netx  = tf.slice(net, [0,0], [s[0]//2,-1])

        print("NETX is ", netx, z)
        if z is not None:
            netx = tf.concat([netx, z], axis=1)

        print("NETX is ", netx)

    return create_g_pyramid_from_z(config, gan, netx, prefix, id, reuse)

def create_g(config, gan, net, prefix="g_", reuse=False):
    with tf.variable_scope("autoencoder", reuse=reuse):
        x_dims = gan.config.x_dims
        z_proj_dims = config.z_projection_depth

        w=int(net.get_shape()[1])
        nets=[]
        activation = config.activation
        batch_size = gan.config.batch_size

        s = [int(x) for x in net.get_shape()]

        if 'align_z' in config:
            w=int(net.get_shape()[1])
            h=int(net.get_shape()[2])
            n2 = linear(gan.graph.z_encoded, w*h, scope=prefix+"lin_pro_zj", gain=config.orthogonal_initializer_gain)
            new_shape = [gan.config.batch_size, w, h, 1]
            n2 = tf.reshape(n2, new_shape)
            net = tf.concat([net, n2], axis=3)
     
        if(config.layer_filter):
            fltr = config.layer_filter(gan, net)
            if(fltr is not None):
                net = tf.concat(axis=3, values=[net, fltr]) # TODO: pass through gan object

        for i in range(config.depth):
            print("_____________________", net)
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

            print("INPUT IS", input,prefix+'laendyers_'+str(i))
            #net = tf.concat([net,input], 3)
            #net = tf.concat([net,original_z], 3)
            name= 'g_layers_end'
            output_channels = (gan.config.channels+(i+1)*6)
            #net = tf.reshape(net, [gan.config.batch_size, primes[0], primes[1], -1])
            if i == config.depth-1:
                output_channels = gan.config.channels
            net = config.block(net, config, activation, batch_size, 'identity', prefix+'laendyers_'+str(i), output_channels=output_channels, filter=3, sigmoid_gate=sigmoid_gate)
            #net = tf.reshape(net, [gan.config.batch_size, primes[0]*4, primes[1]*4, -1])
            first3 = net
            if config.final_activation:
                if config.layer_regularizer:
                    first3 = config.layer_regularizer(gan.config.batch_size, name=prefix+'bn_first3_'+str(i))(first3)
                first3 = config.final_activation(first3)
            if i == config.depth-1:
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



def create_g_pyramid_from_z_2(config, gan, z, prefix="g_", id=0, reuse=False):
    with tf.variable_scope("autoencoder", reuse=reuse):
        gconfig = gan.config.generator_autoencode
        generator = hc.Config(hc.lookup_functions(gconfig))
        bs = int(z.get_shape()[0])
        if id == 0:
            const = tf.one_hot([0], int(gan.graph.y.get_shape()))
            const = tf.reshape(const,[1, -1])
            const = tf.tile(const, [bs, 1])
        else:
            const = gan.graph.y
        z = tf.concat([z, const], 1)
        print("z is ", z)
        rx = generator.create(generator, gan, z, prefix=prefix)[-1]
    
    return rx


def create_g_pyramid_2(config, gan, x, z=None, prefix="g_", id=0, reuse=False):
    with tf.variable_scope("autoencoder", reuse=reuse):
        dconfig = gan.config.discriminators[0]
        dconfig = hc.Config(hc.lookup_functions(dconfig))

        if('align_regularizer' in dconfig):
            dconfig['layer_regularizer'] = config['layer_regularizer']
        g = x
        net = hypergan.discriminators.pyramid_discriminator.discriminator(gan, dconfig, x, g, [x], [g], prefix)
        print("NET is ", net)
        s = [int(x) for x in net.get_shape()]
        netx  = tf.slice(net, [0,0], [s[0]//2,-1])

        print("NETX is ", netx, z)
        if z is not None:
            netx = tf.concat([netx, z], axis=1)

        print("NETX is ", netx)

    return create_g_pyramid_from_z_2(config, gan, netx, prefix, id, reuse)


