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

def text_plot(size, filename, data, x):
    plt.clf()
    plt.figure(figsize=(2,2))
    data = np.squeeze(data)
    plt.plot(x)
    plt.plot(data)
    plt.xlim([0, size])
    plt.ylim([-2, 2.])
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.savefig(filename)

def get_vocabulary():
    lookup_keys = list("~()\"'&+#@/789zyxwvutsrqponmlkjihgfedcba ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456:-,;!?.")
    lookup_values = np.arange(len(lookup_keys), dtype=np.float32)

    lookup = {}

    for i, key in enumerate(lookup_keys):
        lookup[key]=lookup_values[i]

    return lookup_keys, lookup


def parse_args():
    parser = argparse.ArgumentParser(description='Train a 2d test!', add_help=True)
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Examples to include in each batch.  If using batch norm, this needs to be preserved when in server mode')
    parser.add_argument('--device', '-d', type=str, default='/gpu:0', help='In the form "/gpu:0", "/cpu:0", etc.  Always use a GPU (or TPU) to train')
    parser.add_argument('--format', '-f', type=str, default='png', help='jpg or png')
    parser.add_argument('--config', '-c', type=str, default=None, help='config name')
    parser.add_argument('--distribution', '-t', type=str, default='circle', help='what distribution to test, options are circle, modes')
    parser.add_argument('--sample_every', type=int, default=50, help='Samples the model every n epochs.')
    parser.add_argument('--save_every', type=int, default=30000, help='Saves the model every n epochs.')
    return parser.parse_args()

args = parse_args()

def train():
    config = hg.configuration.Configuration.load(args.config + '.json')

    with tf.device(args.device):
        text_input = TextInput(config, args.batch_size, vocabulary = get_vocabulary)
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
        lookup_keys, lookup = get_vocabulary()

        lookup =  {i[1]:i[0] for i in lookup.items()} # reverse hash
        if args.config:
            limit = 10000000
        for i in range(limit):
            gan.step()

            def sample_output(val):
                val *= len(lookup_keys)/2.0
                val += len(lookup_keys)/2.0
                val = np.round(val)

                val = np.maximum(0, val)
                val = np.minimum(len(lookup_keys)-1, val)

                ox_val = [lookup[obj] for obj in list(val)]
                string = "".join(ox_val)
                return string


            if i % args.sample_every == 0 and i > 0:
                g, x_val, reconstruction = gan.session.run([gan.generator.sample, gan.inputs.x, gan.discriminator.reconstruction], {gan.encoder.z: z_0})
                bs = np.shape(x_val)[0]
                x_val = x_val.reshape([bs,-1])
                g = g.reshape([bs,-1])
                reconstruction = reconstruction.reshape([bs,-1])
                sample_file="samples/%06d.png" % (samples)
                #text_plot(64, sample_file, g[0], x_0[0])
                samples+=1
                print("X: "+sample_output(x_val[0]))
                print("RX: "+sample_output(reconstruction[0]))
                print("G:")
                for j, g0 in enumerate(g):
                    if j > 4:
                        break

                    print(sample_output(g0))

            if i % args.save_every == 0 and i > 0 and args.config is not None:
                pass
                #print("Saving " + save_file)
                #with tf.device('/cpu:0'):
                #    gan.save(save_file)


        if args.config is None:
            with open("sequence-results-10k.csv", "a") as myfile:
                myfile.write(config_name+","+str(ax_sum)+","+str(ag_sum)+","+ str(ax_sum+ag_sum)+","+str(ax_sum*ag_sum)+","+str(dlog)+","+str(diversity)+","+str(ax_sum*ag_sum*(1/diversity))+","+str(last_i)+"\n")
        tf.reset_default_graph()
        gan.session.close()

while(True):
    train()
