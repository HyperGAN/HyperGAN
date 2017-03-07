import argparse
import os
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import matplotlib.pyplot as plt
from hypergan.loaders import *
from hypergan.samplers.common import *
from hypergan.util.hc_tf import *
from hypergan.generators import *
import textwrap

def parse_args():
    parser = argparse.ArgumentParser(description='Train a 2d test!', add_help=True)
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Number of samples to include in each batch.  If using batch norm, this needs to be preserved when in server mode')
    parser.add_argument('--device', '-d', type=str, default='/gpu:0', help='In the form "/gpu:0", "/cpu:0", etc.  Always use a GPU (or TPU) to train')
    parser.add_argument('--format', '-f', type=str, default='png', help='jpg or png')
    parser.add_argument('--steps', '-s', type=int, default=10000000, help='number of steps to run for.  defaults to a lot')
    parser.add_argument('--sample_every', type=int, default=50, help='Samples the model every n epochs.')
    parser.add_argument('--config', '-c', type=str, default='2d-test', help='config name')
    parser.add_argument('--save_every', type=int, default=30000, help='Saves the model every n epochs.')
    parser.add_argument('--distribution', '-t', type=str, default='circle', help='what distribution to test, options are circle, modes')
    return parser.parse_args()

def sampler(gan, name):
    generator = gan.graph.g[0]
    z_t = gan.graph.z[0]
    x_t = gan.graph.x
    sess = gan.sess
    config = gan.config
    global x_v
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
    net = linear(net, 16, scope="g_lin_proj3")
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
    gan.config.x_dims = [32,32]
    gan.config.channels = 1
    gs = resize_conv_generator.create(config,gan,net)
    #print("G_S IS", gs)
    #filter = [1,64,4,1]
    #stride = [1,64,4,1]
    #gs[-1] = tf.nn.avg_pool(gs[-1], ksize=filter, strides=stride, padding='SAME')
    #gs[-1] = linear(tf.reshape(gs[-1], [gan.config.batch_size, -1]), 16, scope="g_2d_lin")
    #if config.final_activation:
    #    if config.layer_regularizer:
    #        gs[-1] = config.layer_regularizer(gan.config.batch_size, name='g_bn_first3_')(gs[-1])
    #    gs[-1] = config.final_activation(gs[-1])


    #gs[-1] = tf.slice(gs[-1], [0,16,1,0], [-1, 16, 1, 1])
    gs[-1] = tf.image.resize_images(gs[-1], [16,1], 1)
    gs[-1] = tf.reshape(gs[-1], [gan.config.batch_size, 16])
    return gs

def d_pyramid_create(gan, config, x, g, xs, gs, prefix='d_'):
    print("x,g",x,g)
    with tf.variable_scope("d_input_projection", reuse=False):
        x = linear(x, 32*32, scope=prefix+'input_projection')
        x = tf.reshape(x, [gan.config.batch_size, 32, 32, 1])
    with tf.variable_scope("d_input_projection", reuse=True):
        g = linear(g, 32*32, scope=prefix+'input_projection')
        g = tf.reshape(g, [gan.config.batch_size, 32, 32, 1])
    #x = tf.tile(x, [1, 2*4*32])
    #g = tf.tile(g, [1, 2*4*32])
    #x = tf.reshape(x, [gan.config.batch_size, 64, 64, 1])
    #g = tf.reshape(g, [gan.config.batch_size, 64, 64, 1])
    return hg.discriminators.pyramid_discriminator.discriminator(gan, config, x, g, xs, gs, prefix)


# TODO end shared code

args = parse_args()

selector = hg.config.selector(args)

config = selector.random_config()
config_filename = os.path.expanduser('~/.hypergan/configs/'+args.config+'.json')

custom_config = {
    'model': args.config,
    'batch_size': args.batch_size,
    'generator': g_resize_conv_search_config(),
    'discriminators': [[d_pyramid_search_config()]]
}

custom_selector = hc.Selector()

for key,value in custom_config.items():
    custom_selector.set(key, value)

custom_config = custom_selector.random_config()

for key,value in custom_config.items():
    config[key]=value

config = selector.load_or_create_config(config_filename, config)
print(config)

config = hg.config.lookup_functions(config)
config['dtype']=tf.float32

config['model']=args.config
config['batch_size']=args.batch_size
config['dtype']=tf.float32
config = hg.config.lookup_functions(config)

filename_queue = tf.train.string_input_producer(["/ml/datasets/word2vec/glove.50d.txt"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [["default"]]+[[0.5] for i in range(50)]
name, *wordvecs = tf.decode_csv(
            value, record_defaults=record_defaults, field_delim=' ')
z_wordvecs = tf.stack([wordvecs])
#x=tf.string_split([name], delimiter="")
x=tf.decode_raw(name,tf.uint8)
x=tf.cast(x, tf.float32)
x=(x - 127.5) / 255.0

x = tf.slice(x, [0], [16])

num_preprocess_threads = 8
x, wordvecs = tf.train.shuffle_batch(
  [x, wordvecs],
  batch_size=config.batch_size,
  num_threads=num_preprocess_threads,
  capacity= 1000,
  min_after_dequeue=10)


#with tf.device(args.device):
#    sess = tf.Session()
#    tf.train.start_queue_runners(sess=sess)
#    with sess.as_default():
#        print("Evaling name")
#        name_v, x_v = sess.run([name, x])
#        print(name_v)
#        print(x_v)
#

initial_graph = {
    'x':x,
    'num_labels':1,
}

print("GAN 1 being created..")
save_file = os.path.expanduser("~/.hypergan/saves/"+args.config+".ckpt")
with tf.device(args.device):
    print("GAN being created..", config)
    gan = hg.GAN(config, initial_graph)
    gan.graph.wordvecs_z = wordvecs

    with tf.device('/cpu:0'):
        gan.load_or_initialize_graph(save_file)
    tf.train.start_queue_runners(sess=gan.sess)
    samples = 0

    steps = args.steps
    for i in range(steps):
        d_loss, g_loss = gan.train()

        if i % args.save_every == 0 and i > 0:
            print("Saving " + save_file)
            with(tf.device("/cpu:0")):
                gan.save(save_file)

        if i % args.sample_every == 0 and i > 0:
            g = gan.sess.run(gan.graph.gs[0])
            asciig = np.array(g*255 + 127.5)
            asciig = np.reshape(asciig, [-1])
            s = ""
            for item in asciig:
                try:
                    s+=chr(item)
                except:
                    s+="|"
            print("G IS")
            n=16
            print([s[i:i+n] for i in range(0, len(s), n)])
            samples += 1

    tf.reset_default_graph()
    gan.sess.close()
