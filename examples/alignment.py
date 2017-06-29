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
from hypergan.gans.aligned_gan import AlignedGAN
from common import *

from hypergan.samplers.random_walk_sampler import RandomWalkSampler

arg_parser = ArgumentParser("Align two unaligned distributions.  One is black and white of image.")
arg_parser.add_image_arguments()
args = arg_parser.parse_args()

width, height, channels = parse_size(args.size)

config = lookup_config(args)

save_file = "save/model.ckpt"

if args.action == 'search':
    config = AlignedRandomSearch({}).random_config()

    if args.config_list is not None:
        lines = tuple(open(args.config_list, 'r'))
        config_file = random.choice(lines).strip()
        config = hg.configuration.Configuration.load(config_file+".json")
        random_config = AlignedRandomSearch({}).random_config()

        config["generator"]=random_config["generator"]
        config["discriminator"]=random_config["discriminator"]
        # TODO Other search terms?
        print("config list chosen", config_file)

class TwoImageInput():
    def create(self, args):
        self.inputsa = hg.inputs.image_loader.ImageLoader(args.batch_size)
        self.inputsa.create(args.directory,
                      channels=channels, 
                      format=args.format,
                      crop=args.crop,
                      width=width,
                      height=height,
                      resize=True)


        xa = self.inputsa.x
        xb = tf.tile(tf.image.rgb_to_grayscale(xa), [1,1,1,3])

        self.xa = xa
        self.x = xa #TODO remove
        self.xb = xb

def setup_gan(config, inputs, args):
    gan = AlignedGAN(config=config, inputs=inputs)
    gan.create()

    if(os.path.isfile(save_file+".meta")):
        gan.load(save_file)

    tf.train.start_queue_runners(sess=gan.session)

    GlobalViewer.enable()
    title = "[hypergan] align-test " + args.config
    GlobalViewer.window.set_title(title)

    return gan

def train(config, inputs, args):
    gan = setup_gan(config, inputs, args)
    accuracies = [accuracy(gan.inputs.xb, gan.cycb),accuracy(gan.inputs.xa, gan.cyca)]

    diversities = [batch_diversity(gan.xab), batch_diversity(gan.xba)]

    sampler = AlignedSampler(gan)

    sum_metrics = { 
            "accuracy": [0 for metric in accuracies], 
            "diversity": [0 for metric in diversities] 
    }

    for i in range(args.steps):
        if i % args.sample_every == 0:
            print("Sampling "+str(i))
            sample_file = "samples/"+str(i)+".png"
            sampler.sample(sample_file, args.save_samples)
        gan.step()

        if i % args.save_every == 0 and i > 0:
            print("saving " + save_file)
            gan.save(save_file)

        if i % 10 == 0 and i != 0: 
            if i > 20:
                diversities_v = gan.session.run([v for v in diversities])
                accuracies_v = gan.session.run([v for v in accuracies])
                for k, v in enumerate(diversities_v):
                    if(i > args.steps * 9.0/10):
                        sum_metrics["diversity"][k]+=v
                    if(np.abs(v) < 20000):
                        print("break from diversity")
                        return

                for k, v in enumerate(accuracies_v):
                    if(i > args.steps * 9.0/10):
                        sum_metrics["accuracy"][k]+=v
                    if(np.abs(v) > 800):
                        print("break from accuracy")
                        return
 
    tf.reset_default_graph()
    gan.session.close()
    return sum_metrics

def search(config, inputs, args):
    config_name="alignment-"+str(uuid.uuid4()).split("-")[0]
    config_filename = config_name+'.json'
    print("Saving config to ", config_filename)

    hc.Selector().save(config_filename, config)
    metrics = train(config, inputs, args)

    with open("results-alignment", "a") as myfile:
        accuracies = ["%.2f" % sum for sum in (metrics["accuracy"] or [])]
        diversities = ["%.2f" % sum for sum in (metrics["diversity"] or [])]

        myfile.write(config_name+","+",".join(accuracies)+",".join(diversities)+"\n")

def sample(config, inputs, args):
    gan = setup_gan(config, inputs, args)
    sampler = lookup_sampler(args.sampler or RandomWalkSampler)(gan)
    for i in range(args.steps):
        print("SAMPLER =", sampler)
        sample_file = "samples/"+str(i)+".png"
        sampler.sample(sample_file, False)

inputs = TwoImageInput()
inputs.create(args)

if args.action == 'train':
    metrics = train(config, inputs, args)
    accuracies = ["%.2f" % sum for sum in (metrics["accuracy"] or [])]
    diversities = ["%.2f" % sum for sum in (metrics["diversity"] or [])]

    print("Training complete.  Accuracy", accuracies, "Diversities", diversities)

elif args.action == 'sample':
    sample(config, inputs, args)

elif args.action == 'search':
    search(config, inputs, args)
else:
    print("Unknown action: "+args.action)

if(args.viewer):
    GlobalViewer.window.destroy()
