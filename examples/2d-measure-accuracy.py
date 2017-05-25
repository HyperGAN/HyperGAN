import argparse
import os
import uuid
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import matplotlib.pyplot as plt
from hypergan.loaders import *
from hypergan.util.hc_tf import *
from hypergan.generators import *
from hypergan.search.random_search import RandomSearch

import math

def parse_args():
    parser = argparse.ArgumentParser(description='Train a 2d test!', add_help=True)
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Examples to include in each batch.  If using batch norm, this needs to be preserved when in server mode')
    parser.add_argument('--device', '-d', type=str, default='/gpu:0', help='In the form "/gpu:0", "/cpu:0", etc.  Always use a GPU (or TPU) to train')
    parser.add_argument('--format', '-f', type=str, default='png', help='jpg or png')
    parser.add_argument('--config', '-c', type=str, default='2d-test', help='config name')
    parser.add_argument('--distribution', '-t', type=str, default='circle', help='what distribution to test, options are circle, modes')
    return parser.parse_args()

def no_regularizer(amt):
    return None
 
def l2_distance(a,b):
    return tf.square(a-b)

def l1_distance(a,b):
    return a-b

def custom_discriminator_config():
    selector = hc.Selector()
    selector.set('create', custom_discriminator)
    selector.set('distance',[l2_distance, l1_distance])
    return selector.random_config()

def custom_generator_config():
    return { 
            'create': custom_generator
    }
def custom_encoder_config():
    return { 
            'create': custom_encoder
    }



def custom_discriminator(gan, config, x, g, xs, gs, prefix='d_'):
    net = tf.concat(axis=0, values=[x,g])
    original = net
    net = linear(net, 128, scope=prefix+'linone')
    net = tf.nn.relu(net)
    net = linear(net, 2, scope=prefix+'linend')
    
    # works.  hyperparam? 
    net = config.distance(original,net)
    #net = original-net
    return net

def custom_generator(config, gan, net):
    net = linear(net, 128, scope="g_lin_proj")
    net = tf.nn.relu(net)
    net = linear(net, 2, scope="g_lin_proj3")
    net = tf.tanh(net)
    return [net]



def d_pyramid_search_config():
    return hg.discriminators.pyramid_discriminator.config(
	    activation=[tf.nn.relu, lrelu, tf.nn.relu6, tf.nn.elu],
            depth_increase=[1.5,1.7,2,2.1],
            final_activation=[tf.nn.relu, tf.tanh, None],
            layer_regularizer=[batch_norm_1, layer_norm_1, None],
            layers=[2,1],
            fc_layer_size=[32,16,8,4,2],
            fc_layers=[0,1,2],
            first_conv_size=[4,8,2,1],
            noise=[False, 1e-2],
            progressive_enhancement=[False],
            strided=[True, False],
            create=d_pyramid_create
    )

def g_resize_conv_search_config():
    return resize_conv_generator.config(
            z_projection_depth=[8,16,32],
            activation=[tf.nn.relu,tf.tanh,lrelu,resize_conv_generator.generator_prelu],
            final_activation=[None,tf.nn.tanh,resize_conv_generator.minmax],
            depth_reduction=[2,1.5,2.1],
            layer_filter=None,
            layer_regularizer=[layer_norm_1,batch_norm_1],
            block=[resize_conv_generator.standard_block, resize_conv_generator.inception_block, resize_conv_generator.dense_block],
            resize_image_type=[1],
            create_method=g_resize_conv_create
    )

def g_resize_conv_create(config, gan, net):
    gan.config.x_dims = [8,8]
    gan.config.channels = 1
    gs = resize_conv_generator.create(config,gan,net)
    filter = [1,4,8,1]
    stride = [1,4,8,1]
    gs[0] = tf.nn.avg_pool(gs[0], ksize=filter, strides=stride, padding='SAME')
    #gs[0] = linear(tf.reshape(gs[0], [gan.config.batch_size, -1]), 2, scope="g_2d_lin")
    gs[0] = tf.reshape(gs[0], [gan.config.batch_size, 2])
    return gs

def d_pyramid_create(gan, config, x, g, xs, gs, prefix='d_'):
    with tf.variable_scope("d_input_projection", reuse=False):
        x = linear(x, 8*8, scope=prefix+'input_projection')
        x = tf.reshape(x, [gan.config.batch_size, 8, 8, 1])
    with tf.variable_scope("d_input_projection", reuse=True):
        g = linear(g, 8*8, scope=prefix+'input_projection')
        g = tf.reshape(g, [gan.config.batch_size, 8, 8, 1])
    return hg.discriminators.pyramid_discriminator.discriminator(gan, config, x, g, xs, gs, prefix)

def batch_accuracy(a, b):
    "Each point of a is measured against the closest point on b.  Distance differences are added together."
    tiled_a = a
    tiled_a = tf.reshape(tiled_a, [int(tiled_a.get_shape()[0]), 1, int(tiled_a.get_shape()[1])])

    tiled_a = tf.tile(tiled_a, [1, int(tiled_a.get_shape()[0]), 1])

    tiled_b = b
    tiled_b = tf.reshape(tiled_b, [1, int(tiled_b.get_shape()[0]), int(tiled_b.get_shape()[1])])
    tiled_b = tf.tile(tiled_b, [int(tiled_b.get_shape()[0]), 1, 1])

    difference = tf.abs(tiled_a-tiled_b)
    difference = tf.reduce_min(difference, axis=1)
    difference = tf.reduce_sum(difference, axis=1)
    return tf.reduce_sum(difference, axis=0) 


args = parse_args()

def train():
    savename = "2d-measure-accuracy-"+str(uuid.uuid4())
    savefile = os.path.expanduser('~/.hypergan/configs/'+savename+'.json')

    selector = hc.Selector()

    search = RandomSearch({
        'model': savename,
        'batch_size': args.batch_size,
        'generator': custom_generator_config(),
        'discriminators': [[custom_discriminator_config()]]
        })

    config = search.random_config()

    print("Starting training for: "+savefile)
    selector.save(savefile, config)

    def circle(x):
        spherenet = tf.square(x)
        spherenet = tf.reduce_sum(spherenet, 1)
        lam = tf.sqrt(spherenet)
        return x/tf.reshape(lam,[int(lam.get_shape()[0]), 1])

    def modes(x):
        return tf.round(x*2)/2.0

    if args.distribution == 'circle':
        x = tf.random_normal([args.batch_size, 2])
        x = circle(x)
    elif args.distribution == 'modes':
        x = tf.random_uniform([args.batch_size, 2], -1, 1)
        x = modes(x)
    elif args.distribution == 'sin':
        x = tf.random_uniform((1, args.batch_size), -10.5, 10.5 )
        x = tf.transpose(x)
        r_data = tf.random_normal((args.batch_size,1), mean=0, stddev=0.1)
        xy = tf.sin(0.75*x)*7.0+x*0.5+r_data*1.0
        x = tf.concat([xy,x], 1)/16.0
    elif args.distribution == 'arch':
        offset1 = tf.random_uniform((1, args.batch_size), -10, 10 )
        xa = tf.random_uniform((1, 1), 1, 4 )
        xb = tf.random_uniform((1, 1), 1, 4 )
        x1 = tf.random_uniform((1, args.batch_size), -1, 1 )
        xcos = tf.cos(x1*np.pi + offset1)*xa
        xsin = tf.sin(x1*np.pi + offset1)*xb
        x = tf.transpose(tf.concat([xcos,xsin], 0))/16.0

    initial_graph = {
            'x':x,
            'num_labels':1,
            }


    with tf.device(args.device):
        gan = hg.GAN(config, initial_graph)

        accuracy_x_to_g=batch_accuracy(gan.graph.x, gan.graph.g[0])
        accuracy_g_to_x=batch_accuracy(gan.graph.g[0], gan.graph.x)
        s = [int(g) for g in gan.graph.g[0].get_shape()]
        slice1 = tf.slice(gan.graph.g[0], [0,0], [s[0]//2, -1])
        slice2 = tf.slice(gan.graph.g[0], [s[0]//2,0], [s[0]//2, -1])
        accuracy_g_to_g=batch_accuracy(slice1, slice2)
        x_0 = gan.sess.run(gan.graph.x)
        z_0 = gan.sess.run(gan.graph.z[0])

        gan.initialize_graph()

        ax_sum = 0
        ag_sum = 0
        diversity = 0.00001
        dlog = 0
        last_i = 0

        tf.train.start_queue_runners(sess=gan.sess)
        for i in range(100000):
            d_loss, g_loss = gan.train()

            if i % 500 == 0 and i != 0 and i > 2000: 
                k, ax, ag, agg, dl = gan.sess.run([gan.graph.k, accuracy_x_to_g, accuracy_g_to_x, accuracy_g_to_g, gan.graph.d_log], {gan.graph.x: x_0, gan.graph.z[0]: z_0})

                print("ERROR", ax, ag, k)
                if math.isclose(k, 0.0) or math.isclose(k, 1.0) or np.isnan(g_loss) or np.abs(g_loss) > 50000 or ax > 450:
                    ax_sum = ag_sum = 100000.00
                    print("BEARK")
                    break


            #if(i % 10000 == 0 and i != 0):
            #    g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]
            #    init = tf.initialize_variables(g_vars)
            #    gan.sess.run(init)

            if(i > 95000):
                ax, ag, agg, dl = gan.sess.run([accuracy_x_to_g, accuracy_g_to_x, accuracy_g_to_g, gan.graph.d_log], {gan.graph.x: x_0, gan.graph.z[0]: z_0})
                diversity += agg
                ax_sum += ax
                ag_sum += ag
                dlog = dl

        with open("results.csv", "a") as myfile:
            print("Writing result")
            #measure = gan.sess.run(gan.graph.measure)
            myfile.write(savename+","+str(ax_sum)+","+str(ag_sum)+","+ str(ax_sum+ag_sum)+","+str(ax_sum*ag_sum)+","+str(dlog)+","+str(diversity)+","+str(ax_sum*ag_sum*(1/diversity))+","+str(last_i)+"\n")
        tf.reset_default_graph()
        gan.sess.close()

while(True):
    train()
