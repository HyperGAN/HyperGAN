import argparse
import uuid
import os
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
from hypergan.inputs import *
from hypergan.search.random_search import RandomSearch
from examples.common import CustomGenerator, CustomDiscriminator

def parse_args():
    parser = argparse.ArgumentParser(description='Train an MNIST classifier G(x) = y', add_help=True)
    parser.add_argument('--sample_every', default=500, type=int)
    parser.add_argument('--batch_size', '-b', default=32, type=int)
    parser.add_argument('--device', '-d', type=str, default='/gpu:0', help='In the form "/gpu:0", "/cpu:0", etc.  Always use a GPU (or TPU) to train')
    parser.add_argument('--steps', '-s', type=int, default=40000, help='number of steps to run for.  defaults to a lot')
    return parser.parse_args()

args = parse_args()

class MNISTInputLoader:
    def __init__(self, batch_size):
        from tensorflow.examples.tutorials.mnist import input_data
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        self.x = tf.placeholder(tf.float32, shape=[batch_size, 784])
        self.feed_y = tf.placeholder(tf.float32, shape=[batch_size, 10])
        self.y = ((2*self.feed_y)-1)

while(True):
    savename = "classification-"+str(uuid.uuid4())
    savefile = os.path.expanduser('~/.hypergan/configs/'+savename+'.json')

    search = RandomSearch({
        'generator': {'class': CustomGenerator, 'end_features': 10},
        'discriminator': {'class': CustomDiscriminator}
        })

    config = search.random_config()

    print("Starting training for: "+savefile)

    hc.Selector().save(savefile, config)


    mnist_loader = MNISTInputLoader(args.batch_size)
    gan = hg.GAN(config, inputs=mnist_loader, batch_size=args.batch_size)
    gan.inputs.gradient_penalty_label = gan.inputs.feed_y # TODO: Our X dimensions dont always match the G.  This causes gradient_penalty to fail.
    gan.create()
    mnist = gan.inputs.mnist
    correct_prediction = tf.equal(tf.argmax(gan.generator.sample,1), tf.argmax(gan.inputs.y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

    steps = args.steps
    accuracy_v = 0
    for i in range(steps):
        batch = mnist.train.next_batch(args.batch_size)

        gan.step({gan.inputs.x: batch[0], gan.inputs.feed_y: batch[1]})

        if i % args.sample_every == 0 and i > 0:
            accuracy_v = 0
            repeat_count = 10
            for j in range(repeat_count):
                test_batch = mnist.test.next_batch(args.batch_size)
                accuracy_v += gan.session.run(accuracy,{gan.inputs.x: test_batch[0], gan.inputs.y: test_batch[1]})
            accuracy_v /= repeat_count
            print(accuracy_v)
            batch = mnist.train.next_batch(args.batch_size)

            if(i > 50):
                if(accuracy_v < 10.0):
                    break

    with open("classification-results", "a") as myfile:
        print("Writing result")
        myfile.write(savename+","+str(accuracy_v)+"\n")

    tf.reset_default_graph()
    gan.session.close()
