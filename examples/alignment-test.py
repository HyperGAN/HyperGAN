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

def parse_args():
    parser = argparse.ArgumentParser(description='Train a colorizer!', add_help=True)
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
    parser.add_argument('--add_full_image', type=bool, default=False, help='Instead of just the black and white X, add the whole thing.')
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
                      resize=False)


        xa = self.inputsa.x
        xb = tf.tile(tf.image.rgb_to_grayscale(xa), [1,1,1,3])

        self.xa = xa
        self.xb = xb

class AlignedGAN(BaseGAN):
    def required(self):
        return []

    def create(self):
        config = self.config
        ops = self.ops

        xa = hc.Config({"x":self.inputs.xa})
        xb = hc.Config({"x":self.inputs.xb})
        with tf.variable_scope("ganab"):
            self.gan_atob = StandardGAN(config=config, inputs=xa)
        with tf.variable_scope("ganba"):
            self.gan_btoa = StandardGAN(config=config, inputs=xb)

        self.session = self.ops.new_session(self.ops_config)
        self.gan_atob.session = self.session
        self.gan_btoa.session = self.session

        self.xatoz = self.create_component(config.discriminator)
        self.xbtoz = self.create_component(config.discriminator)

        xaz = self.xatoz.create(x=self.inputs.xa, g=self.inputs.xa)
        xbz = self.xatoz.create(x=self.inputs.xb, g=self.inputs.xb)
        xaz = self.split_batch(xaz)[0]
        xbz = self.split_batch(xbz)[0]

        shape = ops.shape(self.xatoz.sample)
        self.z = tf.random_uniform(shape, config.min or -1, config.max or 1, dtype=ops.dtype)

        def none():
            pass
        def variables():
            return []
        encoder = hc.Config({"sample": self.z, "create": none, "variables": variables})
        self.gan_atob.encoder = encoder
        self.gan_btoa.encoder = encoder

        with tf.variable_scope("ganab"):
            self.gan_atob.create()
        with tf.variable_scope("ganba"):
            self.gan_btoa.create()

        with tf.variable_scope("ganab"):
            self.dga = self.gan_atob.discriminator.reuse(x=self.inputs.xa, g=self.gan_btoa.generator.sample)
        with tf.variable_scope("ganba"):
            self.dgb = self.gan_btoa.discriminator.reuse(x=self.inputs.xb, g=self.gan_atob.generator.sample)

        with tf.variable_scope("ganab"):
            self.cycb = self.gan_atob.generator.reuse(xbz)
        with tf.variable_scope("ganba"):
            self.cyca = self.gan_btoa.generator.reuse(xaz)

        self.cycloss = tf.reduce_mean(tf.abs(self.inputs.xa-self.cyca)) + \
                       tf.reduce_mean(tf.abs(self.inputs.xb-self.cycb))

        self.gan_atob.loss.sample[1] += self.cycloss*10
        self.gan_btoa.loss.sample[1] += self.cycloss*10

        self.gan_atob.trainer = self.gan_atob.create_component(config.trainer)
        self.gan_btoa.trainer = self.gan_btoa.create_component(config.trainer)
        self.gan_atob.trainer.create()
        self.gan_btoa.trainer.create()

        self.session.run(tf.global_variables_initializer())

    def step(self, feed_dict={}):
        self.gan_atob.step(feed_dict)
        self.gan_btoa.step(feed_dict)

two_image_input = TwoImageInput()
two_image_input.create(args)

gan = AlignedGAN(config=config, inputs=two_image_input)
gan.create()

tf.train.start_queue_runners(sess=gan.session)

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

def accuracy(a, b):
    "Each point of a is measured against the closest point on b.  Distance differences are added together."
    difference = tf.abs(a-b)
    difference = tf.reduce_min(difference, axis=1)
    difference = tf.reduce_sum(difference, axis=1)
    return tf.reduce_sum( tf.reduce_sum(difference, axis=0) , axis=0) 

accuracies = {
    #"xa_to_rxa":accuracy(exa, gan.graph.rxa),
    #"xb_to_rxb":accuracy(exb, gan.graph.rxb),
    #"xa_to_rxabba":accuracy(exa, gan.graph.rxabba),
    #"xb_to_rxbaab":accuracy(exb, gan.graph.rxbaab),
    #"xa_to_xabba":accuracy(exa, gan.graph.xabba),
    "xb_to_xab":accuracy(gan.inputs.xb, gan.cycb),
    "xb_to_xba":accuracy(gan.inputs.xa, gan.cyca)
}

diversities={
    #'rxa': batch_diversity(gan.graph.rxa),
    #'rxb': batch_diversity(gan.graph.rxb),
    #'rxabba': batch_diversity(gan.graph.rxabba),
    #'rxbaab': batch_diversity(gan.graph.rxbaab),
    #'rxab': batch_diversity(gan.graph.rxab),
    #'rxba': batch_diversity(gan.graph.rxba),
    'ab': batch_diversity(gan.gan_atob.generator.sample),
    'ba': batch_diversity(gan.gan_btoa.generator.sample)
    #'xabba': batch_diversity(gan.graph.xabba),
    #'xbaab': batch_diversity(gan.graph.xbaab)
}

diversities_items= list(diversities.items())
accuracies_items= list(accuracies.items())

sums = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
names = []


for i in range(40000):
    gan.step()

    if i % 100 == 0 and i != 0: 
        #if 'k' in gan.graph:
        #    k = gan.sess.run([gan.graph.k], {gan.graph.xa: exa, gan.graph.z[0]: static_z})
        #    print("K", k, "d_loss", d_loss)
        #    if math.isclose(k[0], 0.0) or np.isnan(d_loss):
        #        sums = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        #        names = ["error k or dloss"]
        #        break
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

            print("A", accuracies_v)
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
            print(sums)

            if(broken):
                break

        

with open("results-alignment", "a") as myfile:
    myfile.write(config_name+","+",".join(names)+"\n")
    myfile.write(config_name+","+",".join(["%.2f" % sum for sum in sums])+"\n")
 
tf.reset_default_graph()
gan.session.close()
