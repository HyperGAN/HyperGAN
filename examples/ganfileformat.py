import argparse
import os
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
from hypergan.generators import *
from examples.common import CustomDiscriminator, CustomGenerator, Custom2DInputDistribution, Custom2DSampler, Custom2DDiscriminator, Custom2DGenerator
from hypergan.search.random_search import RandomSearch
from hypergan.viewer import GlobalViewer
from examples.common import batch_diversity, accuracy
from hypergan.samplers.batch_sampler import BatchSampler
from hypergan.encoders.base_encoder import BaseEncoder

def parse_args():
    parser = argparse.ArgumentParser(description='project G(None) to single image', add_help=True)
    parser.add_argument('directory', action='store', type=str, help='The location of your data.  Subdirectories are treated as different classes.  You must have at least 1 subdirectory.')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Number of samples to include in each batch.  If using batch norm, this needs to be preserved when in server mode')
    parser.add_argument('--device', '-d', type=str, default='/gpu:0', help='In the form "/gpu:0", "/cpu:0", etc.  Always use a GPU (or TPU) to train')
    parser.add_argument('--format', '-f', type=str, default='png', help='jpg or png')
    parser.add_argument('--size', '-s', type=str, default='64x64x3', help='Size of your data.  For images it is widthxheightxchannels.')
    parser.add_argument('--steps', '-x', type=int, default=10000000, help='number of steps to run for.  defaults to a lot')
    parser.add_argument('--sample_every', type=int, default=50, help='Samples the model every n epochs.')
    parser.add_argument('--config', '-c', type=str, default=None, help='config name')
    parser.add_argument('--distribution', '-t', type=str, default='circle', help='what distribution to test, options are circle, modes')
    return parser.parse_args()


args = parse_args()

config = hg.configuration.Configuration.load(args.config+".json")

GlobalViewer.enable()
config_name = args.config
title = "[hypergan] fileformat " + config_name
GlobalViewer.window.set_title(title)

width = int(args.size.split("x")[0])
height = int(args.size.split("x")[1])
channels = int(args.size.split("x")[2])

class FileGANEncoder(BaseEncoder):
    def create(self):
        self.sample=tf.ones([self.gan.batch_size(), 1])
        self.z = self.sample

with tf.device(args.device):
    input = hg.inputs.image_loader.ImageLoader(args.batch_size)
    input.create(args.directory,
                  channels=channels, 
                  format=args.format,
                  crop=False,
                  width=width,
                  height=height,
                  resize=True)

    gan = hg.GAN(config, inputs = input)
    gan.encoder = FileGANEncoder(gan, {})
    gan.encoder.create()
    gan.create()
    accuracy_t = accuracy(gan.generator.sample, input.x)
    sampler = BatchSampler(gan)

    tf.train.start_queue_runners(sess=gan.session)
    samples = 0
    steps = args.steps
    for i in range(steps):
        gan.step()
        sampler.sample(None, False)

    print("ACCURACY:", gan.session.run(accuracy_t))

    tf.reset_default_graph()
    gan.session.close()
