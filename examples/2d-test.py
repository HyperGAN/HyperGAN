import argparse
import os
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import matplotlib.pyplot as plt
from hypergan.loaders import *
from hypergan.samplers.common import *
from hypergan.util.hc_tf import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train a 2d test!', add_help=True)
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Number of samples to include in each batch.  If using batch norm, this needs to be preserved when in server mode')
    parser.add_argument('--device', '-d', type=str, default='/gpu:0', help='In the form "/gpu:0", "/cpu:0", etc.  Always use a GPU (or TPU) to train')
    parser.add_argument('--format', '-f', type=str, default='png', help='jpg or png')
    parser.add_argument('--steps', '-s', type=int, default=10000000, help='number of steps to run for.  defaults to a lot')
    parser.add_argument('--sample_every', type=int, default=50, help='Samples the model every n epochs.')
    parser.add_argument('--config', '-c', type=str, default='2d-test', help='config name')
    parser.add_argument('--distribution', '-t', type=str, default='circle', help='what distribution to test, options are circle, modes')
    return parser.parse_args()

z_v = None
x_v = None
def sampler(gan, name):
    generator = gan.graph.g[0]
    z_t = gan.graph.z[0]
    x_t = gan.graph.x
    sess = gan.sess
    config = gan.config
    global x_v
    if x_v == None:
        x_v = sess.run(x_t)
    global z_v
    if z_v == None:
        z_v = sess.run(z_t)

    sample = sess.run(generator, {x_t: x_v, z_t: z_v})
    stacks = []
    stacks.append([x_v[1], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7]])
    for i in range(3):
        stacks.append([sample[i*8+8+j] for j in range(8)])
    
    images = np.vstack([np.hstack(s) for s in stacks])
    plt.clf()
    plt.figure(figsize=(5,5))
    plt.scatter(*zip(*x_v), c='b')
    plt.scatter(*zip(*sample), c='r')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.ylabel("z")
    plt.savefig(name)

def no_regularizer(amt):
    return None
 
def custom_discriminator(gan, config, x, g, xs, gs, prefix='d_'):
    net = tf.concat(axis=0, values=[x,g])
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


def custom_discriminator_config(regularizer=no_regularizer, regularizer_lambda=0.0001):
    return { 
            'create': custom_discriminator, 
            'regularizer': regularizer,
            'regularizer_lambda': regularizer_lambda
    }

def custom_generator_config(regularizer=no_regularizer, regularizer_lambda=0.0001):
    return { 
            'create': custom_generator,
            'regularizer': regularizer,
            'regularizer_lambda': regularizer_lambda
    }

args = parse_args()

selector = hg.config.selector(args)

config = selector.random_config()
config_filename = os.path.expanduser('~/.hypergan/configs/'+args.config+'.json')

trainers = []
trainers.append(hg.trainers.adam_trainer.config())

rms_opts = {
    'g_momentum': [0,1e-6],
    'd_momentum': [0,1e-6],
    'd_decay': [0.9,0.99,0.999,0.995],
    'g_decay': [0.9,0.99,0.999,0.995],
    'clipped_gradients': [False, 1e-4],
    'clipped_d_weights': [False, 1e-2],
    'discriminator_learn_rate': [1e-4, 1e-5, 2e-4, 1e-3],
    'generator_learn_rate': [1e-4, 1e-5, 2e-4, 1e-3]
}
trainers.append(hg.trainers.rmsprop_trainer.config(*rms_opts))

custom_config = {
    'model': args.config,
    'batch_size': args.batch_size,
    'trainer': trainers,
    'generator': custom_generator_config(),
    'discriminators': [custom_discriminator_config()]
}

custom_config = hg.config.selector(custom_config).random_config()

for key,value in custom_config.items():
    config[key]=value

config = selector.load_or_create_config(config_filename, config)
config = hg.config.lookup_functions(config)
config['dtype']=tf.float32

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

config['model']=args.config
config['batch_size']=args.batch_size
config['dtype']=tf.float32
config = hg.config.lookup_functions(config)

initial_graph = {
    'x':x,
    'num_labels':1,
}

config['generator']= custom_generator_config()
config['discriminators']= [
        custom_discriminator_config(), 
        custom_discriminator_config()
        ]

with tf.device(args.device):
    gan = hg.GAN(config, initial_graph)
    print("CONFIG", gan.config)

    gan.initialize_graph()
    samples = 0

    tf.train.start_queue_runners(sess=gan.sess)
    steps = args.steps
    for i in range(steps):
        d_loss, g_loss = gan.train()

        if i % args.sample_every == 0 and i > 0:
            print("Sampling "+str(samples))
            sample_file="samples/%06d.png" % (samples)
            gan.sample_to_file(sample_file, sampler=sampler)
            samples += 1

    tf.reset_default_graph()
    sess.close()
