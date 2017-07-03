import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import numpy as np

from hypergan.cli import CLI
from hypergan.gan_component import GANComponent
from hypergan.search.random_search import RandomSearch
from hypergan.generators.base_generator import BaseGenerator
from hypergan.samplers.base_sampler import BaseSampler

class ArgumentParser:
    def __init__(self, description, require_directory=True):
        self.require_directory = require_directory
        self.parser = argparse.ArgumentParser(description=description, add_help=True)
        self.add_global_arguments()
        self.add_search_arguments()
        self.add_train_arguments()

    def add_global_arguments(self):
        parser = self.parser
        parser.add_argument('action', action='store', type=str, help='One of ["train", "search"]')
        if self.require_directory:
            parser.add_argument('directory', action='store', type=str, help='The location of your data.  Subdirectories are treated as different classes.  You must have at least 1 subdirectory.')
        parser.add_argument('--config', '-c', type=str, default='default', help='config name')
        parser.add_argument('--device', '-d', type=str, default='/gpu:0', help='In the form "/gpu:0", "/cpu:0", etc.  Always use a GPU (or TPU) to train')
        parser.add_argument('--batch_size', '-b', type=int, default=32, help='Number of samples to include in each batch.  If using batch norm, this needs to be preserved when in server mode')
        parser.add_argument('--steps', type=int, default=1000000, help='Number of steps to train for.')
        parser.add_argument('--noviewer', dest='viewer', action='store_false', help='Disables the display of samples in a window.')
        parser.add_argument('--save_samples', dest='save_samples', action='store_true', help='Saves samples to disk.')

    def add_search_arguments(self):
        parser = self.parser
        parser.add_argument('--config_list', '-m', type=str, default=None, help='config list name')
        parser.add_argument('--search_output', '-o', type=str, default="search.csv", help='output file for search results')

    def add_train_arguments(self):
        parser = self.parser
        parser.add_argument('--sample_every', type=int, default=50, help='Samples the model every n epochs.')
        parser.add_argument('--save_every', type=int, default=1000, help='Samples the model every n epochs.')

    def add_image_arguments(self):
        parser = self.parser
        parser.add_argument('--crop', type=bool, default=False, help='If your images are perfectly sized you can skip cropping.')
        parser.add_argument('--format', '-f', type=str, default='png', help='jpg or png')
        parser.add_argument('--size', '-s', type=str, default='64x64x3', help='Size of your data.  For images it is widthxheightxchannels.')
        parser.add_argument('--sampler', type=str, default=None, help='Select a sampler.  Some choices: static_batch, batch, grid, progressive')
        return parser

    def parse_args(self):
        return self.parser.parse_args()

class CustomGenerator(BaseGenerator):
    def create(self):
        gan = self.gan
        config = self.config
        ops = self.ops
        print(" CONFIG ", config)
        end_features = config.end_features or 1

        ops.describe('custom_generator')

        net = gan.inputs.x
        net = ops.linear(net, end_features)
        net = ops.lookup('tanh')(net)
        self.sample = net
        return net

class Custom2DGenerator(BaseGenerator):
    def create(self):
        gan = self.gan
        config = self.config
        ops = self.ops
        print(" CONFIG ", config)
        end_features = config.end_features or 1

        ops.describe('custom_generator')

        net = gan.encoder.sample
        net = ops.linear(net, 128)
        net = ops.lookup('relu')(net)
        net = ops.linear(net, end_features)
        net = ops.lookup('tanh')(net)
        self.sample = net
        return net

class CustomDiscriminator(BaseGenerator):
    def build(self, net):
        gan = self.gan
        config = self.config
        ops = self.ops
        ops.describe('custom_discriminator')

        end_features = 1

        x = gan.inputs.x
        y = gan.inputs.y
        g = gan.generator.sample

        print(" CONFIG ", g,y)
        gnet = tf.concat(axis=1, values=[x,g])
        ynet = tf.concat(axis=1, values=[x,y])

        net = tf.concat(axis=0, values=[ynet, gnet])
        net = ops.linear(net, 128)
        net = tf.nn.tanh(net)
        self.sample = net

        return net

class Custom2DDiscriminator(BaseGenerator):
    def create(self):
        gan = self.gan
        x = gan.inputs.x
        g = gan.generator.sample
        net = tf.concat(axis=0, values=[x,g])
        net = self.build(net)
        self.sample = net
        return net

    def build(self, net):
        gan = self.gan
        config = self.config
        ops = self.ops
        ops.describe('custom_discriminator')

        end_features = 1

        net = ops.linear(net, 128)
        net = tf.nn.relu(net)
        net = ops.linear(net, 2)
        self.sample = net

        return net
    def reuse(self, net):
        self.ops.reuse()
        net = self.build(net)
        self.ops.stop_reuse()
        return net 
class Custom2DSampler(BaseSampler):
    def sample(self, filename, save_samples):
        gan = self.gan
        generator = gan.generator.sample

        sess = gan.session
        config = gan.config
        x_v, z_v = sess.run([gan.inputs.x, gan.encoder.z])

        sample = sess.run(generator, {gan.inputs.x: x_v, gan.encoder.z: z_v})

        plt.clf()
        fig = plt.figure(figsize=(3,3))
        plt.scatter(*zip(*x_v), c='b')
        plt.scatter(*zip(*sample), c='r')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.ylabel("z")
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #plt.savefig(filename)
        self.plot(data, filename, save_samples)
        return [{'image': filename, 'label': '2d'}]


class Custom2DInputDistribution:
    def __init__(self, args):
        with tf.device(args.device):
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

            self.x = x
            self.xy = tf.zeros_like(self.x)

def batch_diversity(net):
    bs = int(net.get_shape()[0])
    avg = tf.reduce_mean(net, axis=0)

    s = [int(x) for x in avg.get_shape()]
    avg = tf.reshape(avg, [1, s[0], s[1], s[2]])

    tile = [1 for x in net.get_shape()]
    tile[0] = bs
    avg = tf.tile(avg, tile)
    net -= avg
    return tf.reduce_sum(tf.abs(net))

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

def accuracy(a, b):
    "Each point of a is measured against the closest point on b.  Distance differences are added together."
    difference = tf.abs(a-b)
    difference = tf.reduce_min(difference, axis=1)
    difference = tf.reduce_sum(difference, axis=1)
    return tf.reduce_sum( tf.reduce_sum(difference, axis=0) , axis=0) 

class TextInput:
    def __init__(self, config, batch_size, one_hot=False):
        self.lookup = None
        reader = tf.TextLineReader()
        filename_queue = tf.train.string_input_producer(["chargan.txt"])
        key, x = reader.read(filename_queue)
        vocabulary = self.get_vocabulary()

        table = tf.contrib.lookup.string_to_index_table_from_tensor(
            mapping = vocabulary, default_value = 0)

        x = tf.string_join([x, tf.constant(" " * 64)]) 
        x = tf.substr(x, [0], [64])
        x = tf.string_split(x,delimiter='')
        x = tf.sparse_tensor_to_dense(x, default_value=' ')
        x = tf.reshape(x, [64])
        x = table.lookup(x)
        self.one_hot = one_hot
        if one_hot:
            x = tf.one_hot(x, len(vocabulary))
            x = tf.cast(x, dtype=tf.float32)
            x = tf.reshape(x, [1, int(x.get_shape()[0]), int(x.get_shape()[1]), 1])
        else:
            x = tf.cast(x, dtype=tf.float32)
            x -= len(vocabulary)/2.0
            x /= len(vocabulary)/2.0
            x = tf.reshape(x, [1,1, 64, 1])

        num_preprocess_threads = 8

        x = tf.train.shuffle_batch(
          [x],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity= 512000,
          min_after_dequeue=51200,
          enqueue_many=True)

        self.x = x
        self.table = table

    def get_vocabulary(self):
        vocab = list("~()\"'&+#@/789zyxwvutsrqponmlkjihgfedcba ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456:-,;!?.")
        return vocab

    def np_one_hot(index, length):
        return np.eye(length)[index]

    def get_character(self, data):
        return self.get_lookup_table()[data]

    def get_lookup_table(self):
        if self.lookup is None:
            vocabulary = self.get_vocabulary()
            values = np.arange(len(vocabulary))
            lookup = {}

            if self.one_hot:
                for i, key in enumerate(vocabulary):
                    lookup[key]=self.np_one_hot(values[i], len(values))
            else:
                for i, key in enumerate(vocabulary):
                    lookup[key]=values[i]

            #reverse the hash
            lookup = {i[1]:i[0] for i in lookup.items()}
            self.lookup = lookup
        return self.lookup

    def text_plot(self, size, filename, data, x):
        bs = x.shape[0]
        data = np.reshape(data, [bs, -1])
        x = np.reshape(x, [bs, -1])
        plt.clf()
        plt.figure(figsize=(2,2))
        data = np.squeeze(data)
        plt.plot(x)
        plt.plot(data)
        plt.xlim([0, size])
        plt.ylim([-2, 2.])
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        plt.savefig(filename)

    def sample_output(self, val):
        vocabulary = self.get_vocabulary()
        if self.one_hot:
            vals = [ np.argmax(r) for r in val ]
            ox_val = [vocabulary[obj] for obj in list(vals)]
            string = "".join(ox_val)
            return string
        else:
            val = np.reshape(val, [-1])
            val *= len(vocabulary)/2.0
            val += len(vocabulary)/2.0
            val = np.round(val)

            val = np.maximum(0, val)
            val = np.minimum(len(vocabulary)-1, val)

            ox_val = [self.get_character(obj) for obj in list(val)]
            string = "".join(ox_val)
            return string


def lookup_sampler(name):
    return CLI.sampler_for(name)

def parse_size(size):
    width = int(size.split("x")[0])
    height = int(size.split("x")[1])
    channels = int(size.split("x")[2])
    return [width, height, channels]

def lookup_config(args):
    if args.action == 'train' or args.action == 'sample':
        return hg.configuration.Configuration.load(args.config+".json")
    
def random_config_from_list(config_list_file):
    """ Chooses a random configuration from a list of configs (separated by newline) """
    lines = tuple(open(config_list_file, 'r'))
    config_file = random.choice(lines).strip()
    print("[hypergan] config file chosen from list ", config_list_file, '  file:', config_file)
    return hg.configuration.Configuration.load(config_file+".json")

