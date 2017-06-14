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
from hypergan.samplers.viewer import GlobalViewer
from hypergan.gans.aligned_gan import AlignedGAN

from examples.common import batch_diversity, accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Test alignment!', add_help=True)
    parser.add_argument('directory', action='store', type=str, help='The location of your data.  Subdirectories are treated as different classes.  You must have at least 1 subdirectory.')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Number of samples to include in each batch.  If using batch norm, this needs to be preserved when in server mode')
    parser.add_argument('--crop', type=bool, default=False, help='If your images are perfectly sized you can skip cropping.')
    parser.add_argument('--device', '-d', type=str, default='/gpu:0', help='In the form "/gpu:0", "/cpu:0", etc.  Always use a GPU (or TPU) to train')
    parser.add_argument('--format', '-f', type=str, default='png', help='jpg or png')
    parser.add_argument('--sample_every', type=int, default=50, help='Samples the model every n epochs.')
    parser.add_argument('--size', '-s', type=str, default='64x64x3', help='Size of your data.  For images it is widthxheightxchannels.')
    parser.add_argument('--config', '-c', type=str, default='stable', help='config name')
    parser.add_argument('--config_list', '-m', type=str, default=None, help='config list name')
    parser.add_argument('--use_hc_io', '-9', dest='use_hc_io', action='store_true', help='experimental')
    return parser.parse_args()

args = parse_args()

width = int(args.size.split("x")[0])
height = int(args.size.split("x")[1])
channels = int(args.size.split("x")[2])

config_file = args.config

if args.config_list is not None:
    lines = tuple(open(args.config_list, 'r'))
    config_file = random.choice(lines).strip()
    print("config list chosen", config_file)

config = hg.configuration.Configuration.load(config_file+".json")

config_name="alignment-"+str(uuid.uuid4()).split("-")[0]
config_filename = os.path.expanduser('~/.hypergan/configs/'+config_name+'.json')
print("Saving config to ", config_filename)

hc.Selector().save(config_filename, config)

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

two_image_input = TwoImageInput()
two_image_input.create(args)

gan = AlignedGAN(config=config, inputs=two_image_input)
gan.create()

tf.train.start_queue_runners(sess=gan.session)

accuracies = {
    "xb_to_xab":accuracy(gan.inputs.xb, gan.cycb),
    "xb_to_xba":accuracy(gan.inputs.xa, gan.cyca)
}

diversities={
    'ab': batch_diversity(gan.xab),
    'ba': batch_diversity(gan.xba)
}

diversities_items= list(diversities.items())
accuracies_items= list(accuracies.items())

sums = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
names = []

sampler = AlignedSampler(gan)

GlobalViewer.enable()
config_name = args.config
title = "[hypergan] align-test " + config_name
GlobalViewer.window.set_title(title)

for i in range(40000):
    if i % args.sample_every == 0:
        print("Sampling "+str(i))
        sample_file = "samples/"+str(i)+".png"
        sampler.sample(sample_file)
    gan.step()

    if i % 100 == 0 and i != 0: 
        if i > 200:
            diversities_v = gan.session.run([v for _, v in diversities_items])
            accuracies_v = gan.session.run([v for _, v in accuracies_items])
            broken = False
            for k, v in enumerate(diversities_v):
                sums[k] += v 
                name = diversities_items[k][0]
                names.append(name)
                if(np.abs(v) < 20000):
                    sums = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                    names = ["error diversity "+name]
                    broken = True
                    print("break from diversity")
                    break

            for k, v in enumerate(accuracies_v):
                sums[k+len(diversities_items) ] += v 
                name = accuracies_items[k][0]
                names.append(name)
                if(np.abs(v) > 800):
                    sums = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                    names = ["error accuracy "+name]
                    broken = True
                    print("break from accuracy")
                    break

            if(broken):
                break

with open("results-alignment", "a") as myfile:
    myfile.write(config_name+","+",".join(names)+"\n")
    myfile.write(config_name+","+",".join(["%.2f" % sum for sum in sums])+"\n")
 
tf.reset_default_graph()
gan.session.close()
