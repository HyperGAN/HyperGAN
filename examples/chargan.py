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
from examples.common import *
import numpy as np

import math
import os

arg_parser = ArgumentParser("Learn from a text file", require_directory=False)
arg_parser.parser.add_argument('--one_hot', action='store_true', help='Use character one-hot encodings.')
args = arg_parser.parse_args()

one_hot = args.one_hot

config = lookup_config(args)
if args.action == 'search':
    config = RandomSearch({}).random_config()

def search(config, args):
    metrics = train(config, args)
    config_filename = "chargan-"+str(uuid.uuid4())+'.json'
    hc.Selector().save(config_filename, config)

    with open(args.search_output, "a") as myfile:
        myfile.write(config_filename+","+",".join([str(x) for x in metric_sum])+"\n")

def train(config, args):
    save_file = "save/chargan/model.ckpt"
    with tf.device(args.device):
        text_input = TextInput(config, args.batch_size, one_hot=one_hot)
        gan = hg.GAN(config, inputs=text_input)
        gan.create()

        if(args.action != 'search' and os.path.isfile(save_file+".meta")):
            gan.load(save_file)

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

        vocabulary = text_input.get_vocabulary()

        for i in range(args.steps):
            gan.step()

            if args.action == 'train' and i % args.save_every == 0 and i > 0:
                print("saving " + save_file)
                gan.save(save_file)


            if i % args.sample_every == 0:
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

if args.action == 'train':
    metrics = train(config, args)
    print("Resulting metrics:", metrics)
elif args.action == 'search':
    search(config, args)
else:
    print("Unknown action: "+args.action)
