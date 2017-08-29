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

arg_parser = ArgumentParser("Test your gan vs a known distribution", require_directory=False)
arg_parser.parser.add_argument('--distribution', '-t', type=str, default='circle', help='what distribution to test, options are circle, modes')
args = arg_parser.parse_args()

config = lookup_config(args)
if args.action == 'search':
    config = RandomSearch({}).random_config()
    config['loss']['minibatch'] = False # minibatch breaks on this example


def train(config, args):
    title = "[hypergan] 2d-test " + args.config
    GlobalViewer.title = title
    GlobalViewer.enabled = args.viewer

    with tf.device(args.device):
        config.generator['end_features'] = 2
        gan = hg.GAN(config, inputs = Custom2DInputDistribution(args))
        gan.discriminator = Custom2DDiscriminator(gan, config.discriminator)
        gan.generator = Custom2DGenerator(gan, config.generator)
        gan.encoder = gan.create_component(gan.config.encoder)
        gan.encoder.create()
        gan.generator.create()
        gan.discriminator.create()
        gan.create()

        accuracy_x_to_g=distribution_accuracy(gan.inputs.x, gan.generator.sample)
        accuracy_g_to_x=distribution_accuracy(gan.generator.sample, gan.inputs.x)

        sampler = Custom2DSampler(gan)

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
                for k, metric in enumerate(gan.metrics.keys()):
                    metric_value = gan.session.run(gan.metrics[metric])
                    print("--", metric,  metric_value)
                    if math.isnan(metric_value) or math.isinf(metric_value):
                        print("Breaking due to invalid metric")
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
        search_output = "2d-test-results.csv"

    config_filename = "2d-measure-accuracy-"+str(uuid.uuid4())+'.json'
    hc.Selector().save(config_filename, config)
    with open(search_output, "a") as myfile:
        total = sum(metric_sum)
        myfile.write(config_filename+","+",".join([str(x) for x in metric_sum])+","+str(total)+"\n")
else:
    print("Unknown action: "+args.action)

