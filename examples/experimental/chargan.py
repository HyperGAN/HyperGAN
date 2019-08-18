import argparse
import os
import string
import uuid
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import matplotlib.pyplot as plt
from hypergan.generators import *
from examples.common import *
import numpy as np
from examples.common import *
from hypergan.search.alphagan_random_search import AlphaGANRandomSearch
from hypergan.gans.experimental.alpha_gan import AlphaGAN

arg_parser = ArgumentParser("Learn from a text file", require_directory=False)
arg_parser.parser.add_argument('--one_hot', action='store_true', help='Use character one-hot encodings.')
args = arg_parser.parse_args()


config = lookup_config(args)
if args.action == 'search':
    config = AlphaGANRandomSearch({}).random_config()

def search(config, args):
    metrics = train(config, args)
    config_filename = "chargan-"+str(uuid.uuid4())+'.json'
    hc.Selector().save(config_filename, config)

    with open(args.search_output, "a") as myfile:
        myfile.write(config_filename+","+",".join([str(x) for x in metric_sum])+"\n")


save_file = "save/model.ckpt"

config = lookup_config(args)

inputs = TextInput(config, args.batch_size, one_hot=args.one_hot)

if args.action == 'search':
    random_config = AlphaGANRandomSearch({}).random_config()

    if args.config_list is not None:
        config = random_config_from_list(args.config_list)

        config["generator"]=random_config["generator"]
        config["discriminator"]=random_config["discriminator"]
        # TODO Other search terms?
    else:
        config = random_config


def setup_gan(config, inputs, args):
    gan = hg.GAN(config, inputs=inputs)

    if(args.action != 'search' and os.path.isfile(save_file+".meta")):
        gan.load(save_file)

    with tf.device(args.device):
        with gan.session.as_default():
            inputs.table.init.run()
    tf.train.start_queue_runners(sess=gan.session)

    return gan

def sample(config, inputs, args):
    gan = setup_gan(config, inputs, args)

def search(config, inputs, args):
    gan = setup_gan(config, inputs, args)

def train(config, inputs, args):
    gan = setup_gan(config, inputs, args)

    trainers = []

    with tf.device(args.device):
        s = [int(g) for g in gan.generator.sample.get_shape()]
        x_0 = gan.session.run(gan.inputs.x)
        z_0 = gan.session.run(gan.latent.z)

        ax_sum = 0
        ag_sum = 0
        diversity = 0.00001
        dlog = 0
        last_i = 0
        samples = 0

        vocabulary = inputs.get_vocabulary()

        for i in range(args.steps):
            gan.step()

            if args.action == 'train' and i % args.save_every == 0 and i > 0:
                print("saving " + save_file)
                gan.save(save_file)


            if i % args.sample_every == 0:
                g, x_val = gan.session.run([gan.generator.sample, gan.inputs.x], {gan.latent.z: z_0})
                bs = np.shape(x_val)[0]
                samples+=1
                print("X: "+inputs.sample_output(x_val[0]))
                print("G:")
                for j, g0 in enumerate(g):
                    if j > 4:
                        break

                    print(inputs.sample_output(g0))

        if args.config is None:
            with open("sequence-results-10k.csv", "a") as myfile:
                myfile.write(config_name+","+str(ax_sum)+","+str(ag_sum)+","+ str(ax_sum+ag_sum)+","+str(ax_sum*ag_sum)+","+str(dlog)+","+str(diversity)+","+str(ax_sum*ag_sum*(1/diversity))+","+str(last_i)+"\n")
        tf.reset_default_graph()
        gan.session.close()

if args.action == 'train':
    metrics = train(config, inputs, args)
    print("Resulting metrics:", metrics)
elif args.action == 'sample':
    sample(config, inputs, args)
elif args.action == 'search':
    search(config, inputs, args)
else:
    print("Unknown action: "+args.action)

