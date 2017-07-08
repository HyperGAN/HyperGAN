import argparse
import os
import uuid
import random
import sys
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import numpy as np
from hypergan.generators import *
from hypergan.gans.base_gan import BaseGAN
from hypergan.gans.standard_gan import StandardGAN
from hypergan.samplers.aligned_sampler import AlignedSampler
from hypergan.viewer import GlobalViewer
from hypergan.gans.autoencoder_gan import AutoencoderGAN
from hypergan.search.alphagan_random_search import AlphaGANRandomSearch

from examples.common import *

arg_parser = ArgumentParser("Colorize an image")
arg_parser.add_image_arguments()
args = arg_parser.parse_args()

width, height, channels = parse_size(args.size)

config = lookup_config(args)
if args.action == 'search':
    config = AlphaGANRandomSearch({}).random_config()

save_file = "save/model.ckpt"
inputs = hg.inputs.image_loader.ImageLoader(args.batch_size)
inputs.create(args.directory,
              channels=channels, 
              format=args.format,
              crop=args.crop,
              width=width,
              height=height,
              resize=True)


def setup_gan(config, inputs, args):
    gan = AutoencoderGAN(config=config, inputs=inputs)
    gan.create()

    if(args.action != 'search' and os.path.isfile(save_file+".meta")):
        gan.load(save_file)

 
    tf.train.start_queue_runners(sess=gan.session)
    config_name = args.config
    title = "[hypergan] colorizer " + config_name
    GlobalViewer.title = title

    return gan

def train(config, inputs, args):
    gan = setup_gan(config, inputs, args)
    sampler = lookup_sampler(args.sampler or 'autoencode')(gan)

    metrics = [accuracy(gan.inputs.x, gan.generator.sample), batch_diversity(gan.generator.sample)]
    sum_metrics = [0 for metric in metrics]

    for i in range(args.steps):
        gan.step()

        if i > args.steps * 9.0/10:
            for k, metric in enumerate(gan.session.run(metrics)):
                print("Metric "+str(k)+" "+str(metric))
                sum_metrics[k] += metric 
            
        if i % args.sample_every == 0:
            print("sampling "+str(i))
            sample_file = "samples/"+str(i)+".png"
            sampler.sample(sample_file, args.save_samples)

        if args.action == 'train' and i % args.save_every == 0 and i > 0:
            print("saving " + save_file)
            gan.save(save_file)

    return sum_metrics

def sample(config, inputs, args):
    gan = setup_gan(config, inputs, args)
    sampler = lookup_sampler(args.sampler or 'random_walk')(gan)
    for i in range(args.steps):
        sample_file = "samples/"+str(i)+".png"
        sampler.sample(sample_file, False)

def search(config, inputs, args):
    metrics = train(config, inputs, args)

    config_filename = "autoencode-"+str(uuid.uuid4())+'.json'
    hc.Selector().save(config_filename, config)
    with open(args.search_output, "a") as myfile:
        myfile.write(config_filename+","+",".join([str(x) for x in metrics])+"\n")

if args.action == 'train':
    metrics = train(config, inputs, args)
    print("Resulting metrics:", metrics)
elif args.action == 'sample':
    sample(config, inputs, args)
elif args.action == 'search':
    search(config, inputs, args)
else:
    print("Unknown action: "+args.action)
