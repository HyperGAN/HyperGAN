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
import numpy as np

import math

class TextInput:
    def __init__(self, config, batch_size):
        x = tf.constant("replicate this line 2")
        reader = tf.TextLineReader()
        filename_queue = tf.train.string_input_producer(["chargan.txt"])
        key, line = reader.read(filename_queue)
        x = line
        lookup_keys, lookup = get_vocabulary()
        print("LOOKUP KEYS", lookup_keys)

        table = tf.contrib.lookup.string_to_index_table_from_tensor(
            mapping = lookup_keys, default_value = 0)

        x = tf.string_join([x, tf.constant(" " * 64)]) 
        x = tf.substr(x, [0], [64])
        x = tf.string_split(x,delimiter='')
        x = tf.sparse_tensor_to_dense(x, default_value=' ')
        x = tf.reshape(x, [64])
        x = table.lookup(x)
        x = tf.one_hot(x, len(lookup))
        x = tf.cast(x, dtype=tf.float32)

        print("X___",x.get_shape())
        x = tf.reshape(x, [1, int(x.get_shape()[0]), int(x.get_shape()[1]), 1])
        x = tf.tile(x, [64, 1, 1, 1])
        print("X___",x.get_shape())
        num_preprocess_threads = 8
        
        x = tf.train.shuffle_batch(
          [x],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity= 512000,
          min_after_dequeue=51200,
          enqueue_many=True)

        self.x = x
        self.table = table
            #x=tf.decode_raw(x,tf.uint8)
            #x=tf.cast(x,tf.int32)
            #x = table.lookup(x)
            #x = tf.reshape(x, [64])
            #print("X IS ", x)
            #x = "replicate this line"


            #x=tf.cast(x, tf.float32)
            #x=x / 255.0 * 2 - 1

            #x = tf.constant("replicate this line")


            #--- working manual input ---
            #lookup_keys, lookup = get_vocabulary()

            #input_default = 'reproduce this line                                             '
            #input_default = [lookup[obj] for obj in list(input_default)]
            #
            #input_default = tf.constant(input_default)
            #input_default -= len(lookup_keys)/2.0
            #input_default /= len(lookup_keys)/2.0
            #input_default = tf.reshape(input_default, [1, 64])
            #input_default = tf.tile(input_default, [512, 1])

            #x = tf.placeholder_with_default(
            #        input_default, 
            #        [512, 64])

            #---/ working manual input ---



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

def one_hot(index, length):
    return np.eye(length)[index]

def get_vocabulary():
    lookup_keys = list("~()\"'&+#@/789zyxwvutsrqponmlkjihgfedcba ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456:-,;!?.")
    lookup_values = np.arange(len(lookup_keys))

    lookup = {}

    for i, key in enumerate(lookup_keys):
        lookup[key]=one_hot(lookup_values[i], len(lookup_values))

    return lookup_keys, lookup


def sample_char(v):
    v = v.encode('ascii', errors='ignore')
    print(v)
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

def g_resize_conv_create(config, gan, net):
    gan.config.x_dims = [64,80]
    gan.config.channels = 1
    gs = resize_conv_generator.create(config,gan,net)
    filter = [1,4,8,1]
    stride = [1,4,8,1]
    #gs[0] = tf.nn.avg_pool(gs[0], ksize=filter, strides=stride, padding='SAME')
    #gs[0] = linear(tf.reshape(gs[0], [gan.config.batch_size, -1]), 2, scope="g_2d_lin")
    #gs[-1] = tf.reshape(gs[-1], [gan.config.batch_size, -1])
    print("GS0", gs[-1], gs)
    return gs

def d_pyramid_create(gan, config, x, g, xs, gs, prefix='d_'):
    s = [int(j) for j in x.get_shape()]
    x = tf.reshape(x, [s[0], 64, 80, 1])
    g = tf.reshape(g, [s[0], 64, 80, 1])
    print("XG", x, g)
    return hg.discriminators.pyramid_discriminator.discriminator(gan, config, x, g, xs, gs, prefix)

def batch_accuracy(a, b):
    "Each point of a is measured against the closest point on b.  Distance differences are added together."
    tiled_a = a
    tiled_a = tf.reshape(tiled_a, [int(tiled_a.get_shape()[0]), 1, int(tiled_a.get_shape()[1])])

    tiled_a = tf.tile(tiled_a, [1, int(tiled_a.get_shape()[0]), 1])

    tiled_b = b
    tiled_b = tf.reshape(tiled_b, [1, int(tiled_b.get_shape()[0]), int(tiled_b.get_shape()[1])])
    tiled_b = tf.tile(tiled_b, [int(tiled_b.get_shape()[0]), 1, 1])

    difference = tf.abs(tiled_a-tiled_b)
    difference = tf.reduce_min(difference, axis=1)
    difference = tf.reduce_sum(difference, axis=1)
    return tf.reduce_sum(difference, axis=0) 



def train():
    args = parse_args()
    print( args)
    config_name=args.config

    trainers = []

    config = hg.configuration.Configuration.load(args.config + '.json')


    with tf.device(args.device):
        text_input = TextInput(config, args.batch_size)
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
            #save_file = ...
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
        if args.config:
            limit = 10000000
        for i in range(limit):
            gan.step()

            #if(i % 10000 == 0 and i != 0):
            #    g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]
            #    init = tf.initialize_variables(g_vars)
            #    gan.sess.run(init)

            def sample_output(val):
                vals = [ np.argmax(r) for r in val ]
                ox_val = [lookup_keys[obj] for obj in list(vals)]
                string = "".join(ox_val)
                return string

            if i % args.sample_every == 0 and i > 0:
                g, x_val, reconstruction = gan.session.run([gan.generator.sample, gan.inputs.x, gan.discriminator.reconstruction], {gan.encoder.z: z_0})
                sample_file="samples/%06d.png" % (samples)
                #text_plot(64, sample_file, g[0], x_0[0])
                samples+=1
                lookup_keys, lookup = get_vocabulary()

                print("X:", sample_output(x_val[0]))
                print("RX:", sample_output(reconstruction[0]))

                for j, g_sample in enumerate(g):
                    if j > 4:
                        break

                    print(sample_output(g_sample))

            if i % args.save_every == 0 and i > 0 and args.config is not None:
                print("Saving " + save_file)
                with tf.device('/cpu:0'):
                    gan.save(save_file)


        if args.config is None:
            with open("sequence-results-10k.csv", "a") as myfile:
                myfile.write(config_name+","+str(ax_sum)+","+str(ag_sum)+","+ str(ax_sum+ag_sum)+","+str(ax_sum*ag_sum)+","+str(dlog)+","+str(diversity)+","+str(ax_sum*ag_sum*(1/diversity))+","+str(last_i)+"\n")
        tf.reset_default_graph()
        gan.session.close()

while(True):
    train()
