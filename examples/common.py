import argparse
import hyperchamber as hc
import hypergan as hg
import numpy as np
import random
import torch

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

    shape = a.shape
    tiled_a = a.view(shape[0], 1, shape[1]).repeat(1, shape[0], 1)
    tiled_b = b.view(1, shape[0], shape[1]).repeat(shape[0], 1, 1)

    difference = torch.abs(tiled_a-tiled_b)
    difference = torch.min(difference, dim=1)[0]
    difference = torch.sum(difference, dim=1)
    return torch.sum(difference, dim=0)

def batch_accuracy(a, b):
    "Difference from a to b.  Meant for reconstruction measurements."
    difference = tf.abs(a-b)
    difference = tf.reduce_min(difference, axis=1)
    difference = tf.reduce_sum(difference, axis=1)
    return tf.reduce_sum( tf.reduce_sum(difference, axis=0) , axis=0) 


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

