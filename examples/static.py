import argparse
import os
import uuid
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import numpy as np
from hypergan.generators import *
from hypergan.viewer import GlobalViewer
from common import *
from hypergan.search.random_search import RandomSearch

from hypergan.samplers.random_walk_sampler import RandomWalkSampler
from hypergan.samplers.static_batch_sampler import StaticBatchSampler

arg_parser = ArgumentParser("Feed static values into X/Z and memorize them")
arg_parser.add_image_arguments()
args = arg_parser.parse_args()

width, height, channels = parse_size(args.size)

config = lookup_config(args)

save_file = "save/model.ckpt"

if args.action == 'search':
    config = RandomSearch({}).random_config()

    if args.config_list is not None:
        config = random_config_from_list(args.config_list)
        random_config = RandomSearch({}).random_config()

        config["generator"]=random_config["generator"]
        config["discriminator"]=random_config["discriminator"]
        # TODO Other search terms?

inputs = hg.inputs.image_loader.ImageLoader(args.batch_size)
inputs.create(args.directory,
              channels=channels, 
              format=args.format,
              crop=args.crop,
              width=width,
              height=height,
              resize=True)

save_file = "save/model.ckpt"

def setup_gan(config, inputs, args):
    gan = hg.GAN(config, inputs=inputs)

    gan.create()

    if(args.action != 'search' and os.path.isfile(save_file+".meta")):
        gan.load(save_file)

    tf.train.start_queue_runners(sess=gan.session)

    config_name = args.config
    title = "[hypergan] static " + config_name
    GlobalViewer.title = title

    return gan

def train(config, inputs, args):
    gan = setup_gan(config, inputs, args)
    static_x, static_z = gan.session.run([gan.inputs.x, gan.encoder.sample])

    accuracy_x_to_g=accuracy(static_x, gan.generator.sample)
    diversity_g = batch_diversity(gan.generator.sample)

    metrics = [accuracy_x_to_g, diversity_g]
    sum_metrics = [0 for metric in metrics]
    sampler = lookup_sampler(args.sampler or StaticBatchSampler)(gan)
    for i in range(args.steps):
        gan.step({gan.inputs.x: static_x, gan.encoder.sample: static_z})

        if i % args.sample_every == 0:
            print("sampling "+str(i))
            sample_file = "samples/"+str(i)+".png"
            sampler.sample(sample_file, args.save_samples)

        if args.action == 'train' and i % args.save_every == 0 and i > 0:
            print("saving " + save_file)
            gan.save(save_file)

        if i > args.steps * 9.0/10:
            for k, metric in enumerate(gan.session.run(metrics)):
                print("Metric "+str(k)+" "+str(metric))
                sum_metrics[k] += metric 
    return sum_metrics

def sample(config, inputs, args):
    gan = setup_gan(config, inputs, args)
    sampler = lookup_sampler(args.sampler or RandomWalkSampler)(gan)
    for i in range(args.steps):
        sample_file = "samples/"+str(i)+".png"
        sampler.sample(sample_file, args.save_samples)

def search(config, inputs, args):
    metrics = train(config, inputs, args)
    config_filename = "static-"+str(uuid.uuid4())+'.json'
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
