import argparse
import os
import math

import uuid
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
from hypergan.generators import *
from hypergan.search.random_search import RandomSearch
from common import *

arg_parser = ArgumentParser("Test your gan vs a known distribution", require_directory=False)
arg_parser.parser.add_argument('--distribution', '-t', type=str, default='circle', help='what distribution to test, options are circle, modes')
arg_parser.parser.add_argument('--sequence_length', '-n', type=int, default=2, help='how many steps to look forward')
args = arg_parser.parse_args()

config = lookup_config(args)
if args.action == 'search':
    config = RandomSearch({}).random_config()

class Sequence2DGenerator(BaseGenerator):
    def create(self):
        gan = self.gan
        config = self.config
        ops = self.ops
        end_features = config.end_features or 2*args.sequence_length

        ops.describe('custom_generator')

        net = gan.encoder.sample
        for i in range(2):
            net = ops.linear(net, 32)
            net = ops.lookup('bipolar')(net)
        net = ops.linear(net, end_features)
        print("-- net is ", net)
        self.sample = net
        return net

class Sequence2DDiscriminator(BaseGenerator):
    def __init__(self, gan, config, g=None, x=None, name=None, input=None, reuse=None, features=[], skip_connections=[]):
        self.x = x
        self.g = g

        GANComponent.__init__(self, gan, config, name=name, reuse=reuse)
    def create(self):
        gan = self.gan
        if self.x is None:
            self.x = gan.inputs.x
        if self.g is None:
            self.g = gan.generator.sample
        net = tf.concat(axis=0, values=[self.x,self.g])
        net = self.build(net)
        self.sample = net
        return net

    def build(self, net):
        gan = self.gan
        config = self.config
        ops = self.ops
        layers=2

        end_features = 1

        for i in range(layers):
            net = ops.linear(net, 32)
            net = ops.lookup('bipolar')(net)
        net = ops.linear(net, 1)
        self.sample = net

        return net
    def reuse(self, net):
        self.ops.reuse()
        net = self.build(net)
        self.ops.stop_reuse()
        return net 

class Sequence2DInputDistribution:
    def __init__(self, args):
        self.current_step = tf.Variable(0)

        with tf.device(args.device):
            def circle(step, length=180):
                s = tf.cast(step, tf.float32) / float(length) * np.pi
                x = tf.reshape(tf.sin(s), [1])
                y = tf.reshape(tf.cos(s), [1])
                return tf.concat([x,y], axis=-1)

            if args.distribution == 'circle':
                batch = []
                batch_offset = 39
                for b in range(args.batch_size):
                    result = []
                    for seq in range(args.sequence_length):
                        result += [circle(self.current_step+seq+b*batch_offset)]
                    batch.append([tf.concat(result, axis=-1)])
                x = tf.concat(batch, axis=0)

            elif args.distribution == 'static-point':
                x = tf.ones([args.batch_size, gan.config.sequence_length*2])

            self.x = x

class Sequence2DSampler(BaseSampler):
    def __init__(self, gan):
        self.sample_count = 0
        self.gan = gan
        self.enc = gan.session.run(gan.encoder.sample)
    def sample(self, filename, save_samples):
        gan = self.gan
        generator = gan.generator.sample

        sess = gan.session
        config = gan.config

        x_v, sample = sess.run([gan.inputs.x, generator], {gan.inputs.current_step: self.sample_count, gan.encoder.sample: self.enc})
        self.sample_count+=1

        def diff(xs):
            r =[]
            seq = args.sequence_length
            for i in range(len(xs)-1):
                if (i + 1) % seq == 0:
                    r += [0]
                else:
                    r += [(xs[1+i]-xs[i])[0]]
            r += [0]
            return r

        X, Y = np.split(np.reshape(x_v,[-1,2]), len(np.reshape(x_v,[-1,2])[0]), axis=1)

        V = diff(Y)
        U = diff(X)

        mpl.style.use('classic')
        plt.clf()

        #fig = plt.figure(figsize=(3,3))
        fig = plt.figure()
        plt.scatter(*zip(*np.reshape(x_v,[-1,2])), c='b')
        plt.scatter(*zip(*np.reshape(sample,[-1,2])), c='r')
        q = plt.quiver(X,Y,U,V, color='b', units='x')

        gX, gY = np.split(np.reshape(sample,[-1,2]), len(np.reshape(x_v,[-1,2])[0]), axis=1)

        gV = diff(gY)
        gU = diff(gX)
        q = plt.quiver(gX,gY,gU,gV, color='r', units='x')
        #plt.xlim([-2, 2])
        #plt.ylim([-2, 2])
        #plt.ylabel("z")
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #plt.savefig(filename)
        self.plot(data, filename, save_samples)
        return [{'image': filename, 'label': '2d'}]



def train(config, args):
    title = "[hypergan] 2d-test " + args.config

    with tf.device(args.device):
        config.generator["class"]="class:__main__.Sequence2DGenerator"
        config.discriminator["class"]="class:__main__.Sequence2DDiscriminator"
        gan = hg.GAN(config, inputs = Sequence2DInputDistribution(args))

        sampler = Sequence2DSampler(gan)

        tf.train.start_queue_runners(sess=gan.session)
        samples = 0
        steps = args.steps
        sampler.sample("samples/000000.png", args.save_samples)

        #metrics = [accuracy_x_to_g, accuracy_g_to_x]
        #sum_metrics = [0 for metric in metrics]
        for i in range(steps):
            gan.step({gan.inputs.current_step: i})

            if args.viewer and i % args.sample_every == 0:
                samples += 1
                print("Sampling "+str(samples), args.save_samples)
                sample_file="samples/%06d.png" % (samples)
                sampler.sample(sample_file, args.save_samples)

            #if i > steps * 9.0/10:
            #    for k, metric in enumerate(gan.session.run(metrics)):
            #        sum_metrics[k] += metric 
            #if i % 300 == 0:
            #    for k, metric in enumerate(gan.metrics.keys()):
            #        metric_value = gan.session.run(gan.metrics[metric])
            #        print("--", metric,  metric_value)
            #        if math.isnan(metric_value) or math.isinf(metric_value):
            #            print("Breaking due to invalid metric")
            #            return None

        tf.reset_default_graph()
        gan.session.close()

    return {}#sum_metrics

if args.action == 'train':
    metrics = train(config, args)
    print("Resulting metrics:", metrics)
else:
    print("Unknown action: "+args.action)

