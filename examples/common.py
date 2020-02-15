import argparse
import hypergan as hg
import hyperchamber as hc
import numpy as np
import random

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
        parser.add_argument('--zoom', '-z', type=int, default=1, help='Zoom level')
        parser.add_argument('--sampler', type=str, default=None, help='Select a sampler.  Some choices: static_batch, batch, grid, progressive')
        return parser

    def parse_args(self):
        return self.parser.parse_args()

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

def distribution_accuracy(a, b):
    """
    Each point of a is measured against the closest point on b.  Distance differences are added together.  
    
    This works best on a large batch of small inputs."""
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

def batch_accuracy(a, b):
    "Difference from a to b.  Meant for reconstruction measurements."
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
          capacity= 5000,
          min_after_dequeue=500,
          enqueue_many=True)

        self.x = x
        self.table = table

    def inputs(self):
        return [self.x]
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


def parse_size(size):
    width = int(size.split("x")[0])
    height = int(size.split("x")[1])
    channels = int(size.split("x")[2])
    return [width, height, channels]

def lookup_config(args):
    return hg.configuration.Configuration.load(args.config+".json")
    
def random_config_from_list(config_list_file):
    """ Chooses a random configuration from a list of configs (separated by newline) """
    lines = tuple(open(config_list_file, 'r'))
    config_file = random.choice(lines).strip()
    print("[hypergan] config file chosen from list ", config_list_file, '  file:', config_file)
    return hg.configuration.Configuration.load(config_file+".json")

