import argparse
import uuid
import os
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
from hypergan.inputs import *
from hypergan.search.random_search import RandomSearch
from common import *

arg_parser = ArgumentParser(description='Train an MNIST classifier G(x) = label')
args = arg_parser.parse_args()

class MNISTInputLoader:
    def __init__(self, batch_size):
        from tensorflow.examples.tutorials.mnist import input_data
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        self.x = tf.placeholder(tf.float32, shape=[batch_size, 784])
        self.feed_y = tf.placeholder(tf.float32, shape=[batch_size, 10])
        self.y = ((2*self.feed_y)-1)

config = lookup_config(args)

if args.action == 'search':
    search = RandomSearch({
        'generator': {'class': CustomGenerator, 'end_features': 10},
        'discriminator': {'class': CustomDiscriminator}
        })

    config = search.random_config()

mnist_loader = MNISTInputLoader(args.batch_size)

def setup_gan(config, inputs, args):
    gan = hg.GAN(config, inputs=inputs, batch_size=args.batch_size)
    gan.inputs.gradient_penalty_label = gan.inputs.feed_y # TODO: Our X dimensions dont always match the G.  This causes gradient_penalty to fail.
    gan.create()

    return gan

def train(config, args):
    gan = setup_gan(config, mnist_loader, args)
    correct_prediction = tf.equal(tf.argmax(gan.generator.sample,1), tf.argmax(gan.inputs.y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100
    metrics = [accuracy]
    sum_metrics = [0 for metric in metrics]
    mnist = gan.inputs.mnist
    for i in range(args.steps):
        batch = mnist.train.next_batch(args.batch_size)

        gan.step({gan.inputs.x: batch[0], gan.inputs.feed_y: batch[1]})

        if i % args.sample_every == 0 and i > 0:
            accuracy_v = 0
            repeat_count = 10
            for j in range(repeat_count):
                test_batch = mnist.test.next_batch(args.batch_size)
                accuracy_v += gan.session.run(accuracy,{gan.inputs.x: test_batch[0], gan.inputs.y: test_batch[1]})
            accuracy_v /= repeat_count
            batch = mnist.train.next_batch(args.batch_size)

            if(i > 50):
                if(accuracy_v < 10.0):
                    sum_metrics = [-1 for metric in metrics]
                    break

    return sum_metrics

def search(config, args):
    metrics = train(config, args)
    config_filename = "classification-"+str(uuid.uuid4())+'.json'
    hc.Selector().save(config_filename, config)

    with open(args.search_output, "a") as myfile:
        print("Writing result")
        myfile.write(config_filename+","+",".join([str(x) for x in metrics])+"\n")

if args.action == 'train':
    metrics = train(config, args)
    print("Resulting metrics:", metrics)
elif args.action == 'search':
    search(config, args)
else:
    print("Unknown action: "+args.action)


