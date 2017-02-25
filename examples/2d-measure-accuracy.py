import argparse
import os
import uuid
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import matplotlib.pyplot as plt
from hypergan.loaders import *
from hypergan.util.hc_tf import *

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
 
def custom_discriminator_config():
    return { 
            'create': custom_discriminator ,
            'noise': [1e-2, False]
    }

def custom_generator_config():
    return { 
            'create': custom_generator
    }

def custom_discriminator(gan, config, x, g, xs, gs, prefix='d_'):
    net = tf.concat(axis=0, values=[x,g])
    if(config['noise']):
        net += tf.random_normal(net.get_shape(), mean=0, stddev=config['noise'], dtype=gan.config.dtype)
    net = linear(net, 128, scope=prefix+'lin1')
    net = tf.nn.relu(net)
    net = linear(net, 128, scope=prefix+'lin2')
    return net

def custom_generator(config, gan, net):
    net = linear(net, 128, scope="g_lin_proj")
    net = batch_norm_1(gan.config.batch_size, name='g_bn_1')(net)
    net = tf.nn.relu(net)
    net = linear(net, 2, scope="g_lin_proj3")
    net = tf.tanh(net)
    return [net]


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
    selector = hg.config.selector(args)
    config_name="2d-measure-accuracy-"+str(uuid.uuid4())

    config = selector.random_config()
    config_filename = os.path.expanduser('~/.hypergan/configs/'+config_name+'.json')

    trainers = []

    rms_opts = {
        'g_momentum': [0,0.1,0.01,1e-6,1e-5,1e-1,0.9,0.999, 0.5],
        'd_momentum': [0,0.1,0.01,1e-6,1e-5,1e-1,0.9,0.999, 0.5],
        'd_decay': [0.8, 0.9, 0.99,0.999,0.995,0.9999,1],
        'g_decay': [0.8, 0.9, 0.99,0.999,0.995,0.9999,1],
        'clipped_gradients': [False, 1e-2],
        'clipped_d_weights': [False, 1e-2],
        'd_learn_rate': [1e-3,1e-4,5e-4,1e-6,4e-4, 5e-5],
        'g_learn_rate': [1e-3,1e-4,5e-4,1e-6,4e-4, 5e-5]
    }
    trainers.append(hg.trainers.rmsprop_trainer.config(**rms_opts))

    adam_opts = {}

    adam_opts = {
        'd_learn_rate': [1e-3,1e-4,5e-4,1e-2,1e-6],
        'g_learn_rate': [1e-3,1e-4,5e-4,1e-2,1e-6],
        'd_beta1': [0.9, 0.99, 0.999, 0.1, 0.01, 0.2, 1e-8],
        'd_beta2': [0.9, 0.99, 0.999, 0.1, 0.01, 0.2, 1e-8],
        'g_beta1': [0.9, 0.99, 0.999, 0.1, 0.01, 0.2, 1e-8],
        'g_beta2': [0.9, 0.99, 0.999, 0.1, 0.01, 0.2, 1e-8],
        'd_epsilon': [1e-8, 1, 0.1, 0.5],
        'g_epsilon': [1e-8, 1, 0.1, 0.5],
        'd_clipped_weights': [False, 0.01],
        'clipped_gradients': [False, 0.01]
    }

    trainers.append(hg.trainers.adam_trainer.config(**adam_opts))
    encoders = []

    projections = []
    projections.append([hg.encoders.linear_encoder.modal, hg.encoders.linear_encoder.linear])
    projections.append([hg.encoders.linear_encoder.modal, hg.encoders.linear_encoder.sphere, hg.encoders.linear_encoder.linear])
    projections.append([hg.encoders.linear_encoder.binary, hg.encoders.linear_encoder.sphere])
    projections.append([hg.encoders.linear_encoder.sphere, hg.encoders.linear_encoder.linear])
    projections.append([hg.encoders.linear_encoder.modal, hg.encoders.linear_encoder.sphere])
    projections.append([hg.encoders.linear_encoder.sphere, hg.encoders.linear_encoder.linear, hg.encoders.linear_encoder.gaussian])
    encoder_opts = {
            'z': [16],
            'modes': [2,4,8,16],
            'projections': projections
            }

    losses = []

    loss_opts = {
        'reduce': [tf.reduce_mean,hg.losses.wgan_loss.echo,hg.losses.wgan_loss.linear_projection,tf.reduce_sum,tf.reduce_logsumexp],
        'reverse': [True, False]
    }
    losses.append([hg.losses.wgan_loss.config(**loss_opts)])
    losses.append([hg.losses.lamb_gan_loss.config(**loss_opts)])
    encoders.append([hg.encoders.linear_encoder.config(**encoder_opts)])
    custom_config = {
        'model': args.config,
        'batch_size': args.batch_size,
        'trainer': trainers,
        'generator': custom_generator_config(),
        'discriminators': [[custom_discriminator_config()]],
        'losses': losses,
        'encoders': encoders
    }

    custom_config_selector = hc.Selector()
    for key,value in custom_config.items():
        custom_config_selector.set(key, value)
        print("Set ", key, value)
    
    custom_config_selection = custom_config_selector.random_config()

    for key,value in custom_config_selection.items():
        config[key]=value

    
    config['dtype']=tf.float32
    config = hg.config.lookup_functions(config)

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

    selector.save(config_filename, config)

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
        for i in range(20000):
            d_loss, g_loss = gan.train()

            if(np.abs(d_loss) > 1000 or np.abs(g_loss) > 1000):
                ax_sum = ag_sum = 100000.00
                break

            #if(i % 10000 == 0 and i != 0):
            #    g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]
            #    init = tf.initialize_variables(g_vars)
            #    gan.sess.run(init)

            if(i > 19000):
                ax, ag, agg, dl = gan.sess.run([accuracy_x_to_g, accuracy_g_to_x, accuracy_g_to_g, gan.graph.d_log], {gan.graph.x: x_0, gan.graph.z[0]: z_0})
                diversity += agg
                ax_sum += ax
                ag_sum += ag
                dlog = dl

        with open("results.csv", "a") as myfile:
            myfile.write(config_name+","+str(ax_sum)+","+str(ag_sum)+","+ str(ax_sum+ag_sum)+","+str(ax_sum*ag_sum)+","+str(dlog)+","+str(diversity)+","+str(ax_sum*ag_sum*(1/diversity))+","+str(last_i)+"\n")
        tf.reset_default_graph()
        gan.sess.close()

while(True):
    train()
