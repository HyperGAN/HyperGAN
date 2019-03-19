import argparse
import uuid
import os
import sys
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

        self.x = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])
        self.feed_y = tf.placeholder(tf.float32, shape=[batch_size, 10])
        self.y = ((2*self.feed_y)-1)

    def layer(self, name):
        return getattr(self, name)

class MNISTGenerator(BaseGenerator):
    def create(self):
        gan = self.gan
        config = self.config
        ops = self.ops
        end_features = config.end_features or 10

        ops.describe('custom_generator')

        net = gan.inputs.x
        net = ops.reshape(net, [gan.batch_size(), -1])
        net = ops.linear(net, end_features)
        net = ops.lookup('tanh')(net)
        self.fy = net
        self.sample = net
        return net
    def layer(self, name):
        return getattr(self, name)
class MNISTDiscriminator(BaseGenerator):
    def build(self, net):
        gan = self.gan
        config = self.config
        ops = self.ops

        end_features = 1

        x = gan.inputs.x
        y = gan.inputs.y
        x = ops.reshape(x, [gan.batch_size(), -1])
        g = gan.generator.sample

        print("G", x, g)
        gnet = tf.concat(axis=1, values=[x,g])
        ynet = tf.concat(axis=1, values=[x,y])

        net = tf.concat(axis=0, values=[ynet, gnet])
        net = ops.linear(net, 128)
        net = tf.nn.tanh(net)
        self.sample = net

        return net



config = lookup_config(args)

if args.action == 'search':
    search = RandomSearch({
        'generator': {'class': MNISTGenerator, 'end_features': 10},
        'discriminator': {'class': MNISTDiscriminator}
        })

    config = search.random_config()

mnist_loader = MNISTInputLoader(args.batch_size)

def setup_gan(config, inputs, args):
    gan = hg.GAN(config, inputs=inputs, batch_size=args.batch_size)
    return gan

def train(config, args):
    gan = setup_gan(config, mnist_loader, args)
    correct_prediction = tf.equal(tf.argmax(gan.generator.layer('fy'),1), tf.argmax(gan.inputs.y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100
    metrics = [accuracy]
    sum_metrics = [0 for metric in metrics]
    mnist = gan.inputs.mnist

    test_batches = []
    test_batch_image = []
    test_batch_label = []
    for _i, _y in zip(mnist.test._images, mnist.test._labels):
        test_batch_image += [_i]
        test_batch_label += [_y]
        if gan.batch_size() == len(test_batch_label):
            test_batches.append([test_batch_image, test_batch_label])
            test_batch_image = []
            test_batch_label = []
    print(str(len(test_batch_label)) + " tests excluded because of batch size")

    for i in range(args.steps):
        batch = mnist.train.next_batch(args.batch_size)
        input_x = np.reshape(batch[0], [gan.batch_size(), 28, 28, 1])

        gan.step({gan.inputs.x: input_x, gan.inputs.feed_y: batch[1]})

        if i % args.sample_every == 0 and i > 0:
            accuracy_v = 0
            test_batch = mnist.test.next_batch(args.batch_size)
            for test_batch in test_batches:
                input_x = np.reshape(test_batch[0], [gan.batch_size(), 28, 28, 1])
                accuracy_v += gan.session.run(accuracy,{gan.inputs.x: input_x, gan.inputs.y: test_batch[1]})
            accuracy_v /= len(test_batches) 
            print("accuracy: ", accuracy_v)

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


