import argparse
import os
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
from hypergan.generators import *
from examples.common import CustomDiscriminator, CustomGenerator, Custom2DInputDistribution, Custom2DSampler, Custom2DDiscriminator, Custom2DGenerator
from hypergan.search.random_search import RandomSearch
from hypergan.viewer import GlobalViewer

def parse_args():
    parser = argparse.ArgumentParser(description='Train a 2d test!', add_help=True)
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Number of samples to include in each batch.  If using batch norm, this needs to be preserved when in server mode')
    parser.add_argument('--device', '-d', type=str, default='/gpu:0', help='In the form "/gpu:0", "/cpu:0", etc.  Always use a GPU (or TPU) to train')
    parser.add_argument('--format', '-f', type=str, default='png', help='jpg or png')
    parser.add_argument('--steps', '-s', type=int, default=10000000, help='number of steps to run for.  defaults to a lot')
    parser.add_argument('--sample_every', type=int, default=50, help='Samples the model every n epochs.')
    parser.add_argument('--config', '-c', type=str, default=None, help='config name')
    parser.add_argument('--distribution', '-t', type=str, default='circle', help='what distribution to test, options are circle, modes')
    return parser.parse_args()


args = parse_args()

config_filename = os.path.expanduser('~/.hypergan/configs/'+args.config+'.json')
config = hc.Selector().load(config_filename)

with tf.device(args.device):
    config.generator['end_features'] = 2
    gan = hg.GAN(config, inputs = Custom2DInputDistribution(args))
    gan.discriminator = Custom2DDiscriminator(gan, config.discriminator)
    gan.generator = Custom2DGenerator(gan, config.generator)
    gan.create()
    print("CONFIG", gan.config)

    sampler = Custom2DSampler(gan)

    tf.train.start_queue_runners(sess=gan.session)
    samples = 0
    steps = args.steps
    sampler.sample("samples/000000.png")
    for i in range(steps):
        gan.step()

        if i % args.sample_every == 0 and i > 0:
            samples += 1
            print("Sampling "+str(samples))
            sample_file="samples/%06d.png" % (samples)
            sampler.sample(sample_file)


    tf.reset_default_graph()
    gan.session.close()
