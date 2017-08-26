import argparse
import os
import math

import uuid
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
from hypergan.generators import *
from hypergan.search.random_search import RandomSearch
from hypergan.viewer import GlobalViewer
from common import *

from hypergan.gan_component import GANComponent
from hypergan.search.random_search import RandomSearch
from hypergan.generators.base_generator import BaseGenerator
from hypergan.discriminators.base_discriminator import BaseDiscriminator
from hypergan.samplers.base_sampler import BaseSampler


class Custom1DGenerator(BaseGenerator):
    def create(self):
        gan = self.gan
        config = self.config
        ops = self.ops
        end_features = config.end_features or 1

        ops.describe('custom_generator')

        net = gan.encoder.sample
        net = ops.linear(net, 128)
        net = ops.lookup('relu')(net)
        net = ops.linear(net, end_features)
        self.sample = net
        return net


class Custom1DDiscriminator(BaseDiscriminator):
    def create(self):
        gan = self.gan
        x = gan.inputs.x
        g = gan.generator.sample
        print("NNET IS", x, g)
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
        net = ops.linear(net, 1)
        self.sample = net

        return net
    def reuse(self, net):
        self.ops.reuse()
        net = self.build(net)
        self.ops.stop_reuse()
        return net 

class Custom1DSampler(BaseSampler):
    def sample(self, filename, save_samples):
        gan = self.gan
        generator = gan.generator.sample

        sess = gan.session
        config = gan.config
        x_v, z_v = sess.run([gan.inputs.x, gan.encoder.z])

        sample = sess.run(generator, {gan.inputs.x: x_v, gan.encoder.z: z_v})

        plt.clf()
        fig = plt.figure(figsize=(3,3))
        n, bins, patches = plt.hist(x_v, 100, facecolor='blue', alpha=0.5)

        plt.plot(bins)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])

        #plt.hist(*zip(*sample), normed=True, c='r')
        #plt.ylim([-2, 2])
        #plt.ylabel("z")
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #plt.savefig(filename)
        self.plot(data, filename, save_samples)
        return [{'image': filename, 'label': '2d'}]


class Custom1DInputDistribution:
    def __init__(self, args):
        with tf.device(args.device):
            distribution = args.distribution or 'gm'

            #if distribution == 'gm':
            x = tf.random_normal([args.batch_size, 1], mean=1, stddev=0.5)
            x += tf.random_normal([args.batch_size, 1], mean=-1, stddev=0.5)
            self.x = x


arg_parser = ArgumentParser("Test your gan vs a known distribution", require_directory=False)
arg_parser.parser.add_argument('--distribution', '-t', type=str, default='circle', help='what distribution to test, options are circle, modes')
args = arg_parser.parse_args()

config = lookup_config(args)
if args.action == 'search':
    config = RandomSearch({}).random_config()
    config['loss']['minibatch'] = False # minibatch breaks on this example


def train(config, args):
    title = "[hypergan] 1d-test " + args.config
    GlobalViewer.title = title
    GlobalViewer.enabled = args.viewer

    with tf.device(args.device):
        config.generator['end_features'] = 1
        gan = hg.GAN(config, inputs = Custom1DInputDistribution(args))
        gan.discriminator = Custom1DDiscriminator(gan, config.discriminator)
        gan.generator = Custom1DGenerator(gan, config.generator)
        gan.encoder = gan.create_component(gan.config.encoder)
        gan.encoder.create()
        gan.generator.create()
        gan.discriminator.create()
        gan.create()

        accuracy_x_to_g=batch_accuracy(gan.inputs.x, gan.generator.sample)
        accuracy_g_to_x=batch_accuracy(gan.generator.sample, gan.inputs.x)

        sampler = Custom1DSampler(gan)

        tf.train.start_queue_runners(sess=gan.session)
        samples = 0
        steps = args.steps
        sampler.sample("samples/000000.png", args.save_samples)

        metrics = [accuracy_x_to_g, accuracy_g_to_x]
        sum_metrics = [0 for metric in metrics]
        for i in range(steps):
            gan.step()

            if args.viewer and i % args.sample_every == 0:
                samples += 1
                print("Sampling "+str(samples), args.save_samples)
                sample_file="samples/%06d.png" % (samples)
                sampler.sample(sample_file, args.save_samples)

            if i > steps * 9.0/10:
                for k, metric in enumerate(gan.session.run(metrics)):
                    sum_metrics[k] += metric 
            if i % 300 == 0:
                print("Checking")
                for k, metric in enumerate(gan.metrics.keys()):
                    if metric== 'gradient_penalty':
                        print("--", gan.session.run(gan.metrics[metric]))
                        if math.isnan(gan.session.run(gan.metrics[metric])):
                            return None

        tf.reset_default_graph()
        gan.session.close()

    return sum_metrics

if args.action == 'train':
    metrics = train(config, args)
    print("Resulting metrics:", metrics)
elif args.action == 'search':
    metric_sum = train(config, args)
    if 'search_output' in args:
        search_output = args.search_output
    else:
        search_output = "1d-test-results.csv"

    config_filename = "1d-measure-accuracy-"+str(uuid.uuid4())+'.json'
    hc.Selector().save(config_filename, config)
    with open(search_output, "a") as myfile:
        total = sum(metric_sum)
        myfile.write(config_filename+","+",".join([str(x) for x in metric_sum])+","+str(total)+"\n")
else:
    print("Unknown action: "+args.action)

