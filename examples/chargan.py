import argparse
import os
import string
import uuid
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import matplotlib.pyplot as plt
from hypergan.loaders import *
from hypergan.generators import *
from examples.common import TextInput
import numpy as np

import math

def parse_args():
    parser = argparse.ArgumentParser(description='Train a sequence of characters using modes (real valued) output.  Related to chargan', add_help=True)
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Examples to include in each batch.  If using batch norm, this needs to be preserved when in server mode')
    parser.add_argument('--device', '-d', type=str, default='/gpu:0', help='In the form "/gpu:0", "/cpu:0", etc.  Always use a GPU (or TPU) to train')
    parser.add_argument('--format', '-f', type=str, default='png', help='jpg or png')
    parser.add_argument('--config', '-c', type=str, default=None, help='config name')
    parser.add_argument('--distribution', '-t', type=str, default='circle', help='what distribution to test, options are circle, modes')
    parser.add_argument('--sample_every', type=int, default=50, help='Samples the model every n epochs.')
    parser.add_argument('--save_every', type=int, default=30000, help='Saves the model every n epochs.')
    parser.add_argument('--one_hot', action='store_true', help='Use character one-hot encodings.')
    return parser.parse_args()

args = parse_args()
one_hot = args.one_hot

def train():
    config = hg.configuration.Configuration.load(args.config + '.json')

    with tf.device(args.device):
        text_input = TextInput(config, args.batch_size, one_hot=one_hot)
        gan = hg.GAN(config, inputs=text_input)
        gan.create()

        with gan.session.as_default():
            text_input.table.init.run()
        tf.train.start_queue_runners(sess=gan.session)

        s = [int(g) for g in gan.generator.sample.get_shape()]
        x_0 = gan.session.run(gan.inputs.x)
        z_0 = gan.session.run(gan.encoder.z)

        if args.config is not None:
            pass
            #save_file = os.path.expanduser("~/.hypergan/saves/"+args.config+".ckpt")
            #with tf.device('/cpu:0'):
            #    gan.load_or_initialize_graph(save_file)
        else:
            save_file = None
            #gan.initialize_graph()

        ax_sum = 0
        ag_sum = 0
        diversity = 0.00001
        dlog = 0
        last_i = 0
        samples = 0

        tf.train.start_queue_runners(sess=gan.session)

        limit = 10000
        vocabulary = text_input.get_vocabulary()

        if args.config:
            limit = 10000000
        for i in range(limit):
            gan.step()


            if i % args.sample_every == 0 and i > 0:
                g, x_val = gan.session.run([gan.generator.sample, gan.inputs.x], {gan.encoder.z: z_0})
                bs = np.shape(x_val)[0]
                samples+=1
                print("X: "+text_input.sample_output(x_val[0]))
                print("G:")
                for j, g0 in enumerate(g):
                    if j > 4:
                        break

                    print(text_input.sample_output(g0))

        if args.config is None:
            with open("sequence-results-10k.csv", "a") as myfile:
                myfile.write(config_name+","+str(ax_sum)+","+str(ag_sum)+","+ str(ax_sum+ag_sum)+","+str(ax_sum*ag_sum)+","+str(dlog)+","+str(diversity)+","+str(ax_sum*ag_sum*(1/diversity))+","+str(last_i)+"\n")
        tf.reset_default_graph()
        gan.session.close()

while(True):
    train()
