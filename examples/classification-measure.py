import argparse
import uuid
import os
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
from hypergan.loaders import *
from hypergan.samplers.common import *
from hypergan.util.hc_tf import *
from hypergan.generators import *
from hypergan.search.random_search import RandomSearch

def parse_args():
    parser = argparse.ArgumentParser(description='Train a 2d test!', add_help=True)
    parser.add_argument('--sample_every', default=500, type=int)
    parser.add_argument('--batch_size', '-b', default=32, type=int)
    parser.add_argument('--device', '-d', type=str, default='/gpu:0', help='In the form "/gpu:0", "/cpu:0", etc.  Always use a GPU (or TPU) to train')
    parser.add_argument('--steps', '-s', type=int, default=40000, help='number of steps to run for.  defaults to a lot')
    return parser.parse_args()

def custom_discriminator(gan, config, x, g, xs, gs, prefix='d_'):
    end_features = 1

    gnet = tf.concat(axis=1, values=[x,g])
    xynet = tf.concat(axis=1, values=[x,gan.graph.xy])
    net = tf.concat(axis=0, values=[xynet, gnet])
    net = linear(net, 1024, scope=prefix+'xlinone')
    net = tf.nn.relu(net)
    net = linear(net, end_features, scope=prefix+'linend')
    net = tf.nn.tanh(net)

    return net

def custom_generator(config, gan, net):
    end_features = ((int)(gan.graph.xy.get_shape()[1]))
    net = gan.graph.x
    net = linear(net, end_features, scope="g_lin_proj3")
    net = tf.tanh(net)
    return [net]

def custom_discriminator_config():
    return { 
            'create': custom_discriminator
    }

def custom_generator_config():
    return { 
            'create': custom_generator
    }

x_v = None
xy_v = None
def sampler(gan, name):
    generator = gan.graph.g[0]
    x_t = gan.graph.x
    xy_t = gan.graph.oy
    sess = gan.sess
    config = gan.config
    global x_v, xy_v
    if x_v == None:
        batch = mnist.train.next_batch(50)
        x_v = batch[0]
        xy_v = batch[1]

    xy, sample = sess.run([gan.graph.xy,generator], {x_t: x_v, xy_t: xy_v})
    print("SAMPLE", sample[0], xy[0])
    plt.clf()
    plt.figure(figsize=(5,5))
    plt.plot(sample[0])
    plt.plot(xy[0])
    plt.xlim([0, 9])
    plt.ylim([-1,1])
    plt.savefig(name)

args = parse_args()

while(True):
    savename = "classification-"+str(uuid.uuid4())
    savefile = os.path.expanduser('~/.hypergan/configs/'+savename+'.json')

    selector = hc.Selector()

    search = RandomSearch({
        'model': savename,
        'batch_size': args.batch_size,
        'generator': custom_generator_config(),
        'discriminators': [[custom_discriminator_config()]]
        })

    config = search.random_config()

    print("Starting training for: "+savefile)

    config['batch_size']=args.batch_size
    config['dtype']=tf.float32
    config = hg.config.lookup_functions(config)

    selector.save(savefile, config)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])

    initial_graph = {
        'x':x,
        'oy': y,
        'xy':((2*y)-1)
    }

    with tf.device(args.device):

        gan = hg.GAN(config, initial_graph)
        correct_prediction = tf.equal(tf.argmax(gan.graph.g[0],1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100
        print("CONFIG", gan.config)

        gan.initialize_graph()
        samples = 0

        steps = args.steps
        accuracy_v = 0
        for i in range(steps):
            batch = mnist.train.next_batch(50)

            d_loss, g_loss = gan.train({x: batch[0], y: batch[1]})

            if i % args.sample_every == 0 and i > 0:
                accuracy_v = gan.sess.run(accuracy,{x: mnist.test.images, y: mnist.test.labels})
                print(accuracy_v)
                batch = mnist.train.next_batch(50)
                g_v,y_v = gan.sess.run([gan.graph.g, gan.graph.xy], {x: batch[0], y: batch[1]})

                print("Sampling "+str(samples))
                if(samples > 3):
                    if(accuracy_v < 10.0):
                        break
                sample_file="samples/%06d.png" % (samples)
                gan.sample_to_file(sample_file, sampler=sampler)
                samples += 1

        with open("classification-results", "a") as myfile:
            print("Writing result")
            myfile.write(savename+","+str(accuracy_v)+"\n")

        tf.reset_default_graph()
        gan.sess.close()
