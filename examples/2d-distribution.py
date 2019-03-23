import argparse
import io
import os
import math

import uuid
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import json
from hypergan.generators import *
from hypergan.search.random_search import RandomSearch
from hypergan.viewer import GlobalViewer
from common import *
from PIL import Image
import plotly.graph_objs as go

arg_parser = ArgumentParser("Test your gan vs a known distribution", require_directory=False)
arg_parser.parser.add_argument('--distribution', '-t', type=str, default='circle', help='what distribution to test, options are circle, modes')
arg_parser.parser.add_argument('--contour_size', '-cs', type=int, default=128, help='number of points to plot the discriminator contour with.  must be a multiple of batch_size')
arg_parser.parser.add_argument('--sample_points', '-p', type=int, default=512, help='number of scatter points to plot.  must be a multiple of batch_size')
args = arg_parser.parse_args()

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.io as pio

class Custom2DDiscriminator(BaseGenerator):
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
            net = ops.linear(net, 16)
            net = ops.lookup('bipolar')(net)
        net = ops.linear(net, 1)
        self.sample = net

        return net


class Custom2DGenerator(BaseGenerator):
    def create(self):
        gan = self.gan
        config = self.config
        ops = self.ops
        end_features = config.end_features or 1

        ops.describe('custom_generator')

        net = gan.latent.sample
        for i in range(2):
            net = ops.linear(net, 16)
            net = ops.lookup('bipolar')(net)
        net = ops.linear(net, end_features)
        print("-- net is ", net)
        self.sample = net
        return net

class Custom2DInputDistribution:
    def __init__(self, args):
        with tf.device(args.device):
            def circle(x):
                spherenet = tf.square(x)
                spherenet = tf.reduce_sum(spherenet, 1)
                lam = tf.sqrt(spherenet)
                return x/tf.reshape(lam,[int(lam.get_shape()[0]), 1])

            def modes(x):
                shape = x.get_shape()
                return tf.round(x*2)/2.0#+tf.random_normal(shape, 0, 0.04)

            if args.distribution == 'circle':
                x = tf.random_normal([args.batch_size, 2])
                x = circle(x)
            elif args.distribution == 'modes':
                x = tf.random_uniform([args.batch_size, 2], -1, 1)
                x = modes(x)
            elif args.distribution == 'modal-gaussian':
                x = tf.random_uniform([args.batch_size, 2], -1, 1)
                y = tf.random_normal([args.batch_size, 2], stddev=0.04, mean=0.15)
                x = tf.round(x) + y
            elif args.distribution == 'sin':
                x = tf.random_uniform((1, args.batch_size), -10.5, 10.5 )
                x = tf.transpose(x)
                r_data = tf.random_normal((args.batch_size,1), mean=0, stddev=0.1)
                xy = tf.sin(0.75*x)*7.0+x*0.5+r_data*1.0
                x = tf.concat([xy,x], 1)/16.0

            elif args.distribution == 'static-point':
                x = tf.ones([args.batch_size, 2])

            self.x = x
            self.xy = tf.zeros_like(self.x)


x_v, z_v = None, None
class Custom2DSampler(BaseSampler):
    def __init__(self, gan):
        self.gan = gan
        self.copy_vars = [tf.Variable(x) for x in self.gan.variables()]
        self.reset_vars = [y.assign(x) for y, x in zip(self.copy_vars, self.gan.variables())]

    def sample(self, filename, save_samples):
        gan = self.gan
        generator = gan.generator.sample

        sess = gan.session
        config = gan.config

        contours = args.contour_size

        x,y = np.meshgrid(np.arange(-1.5, 1.5, 3/contours), np.arange(-1.5, 1.5, 3/contours))
        d = []
        for i in range(args.contour_size):
            _x = np.reshape(x[:,i], [-1]) 
            _y = np.reshape(y[:,i], [-1]) 
            for j in range(args.contour_size // gan.batch_size()):
                offset = j*gan.batch_size()
                endoffset = (j+1)*gan.batch_size()
                _x_sample = _x[offset:endoffset]
                _y_sample = _y[offset:endoffset]
                _d = gan.session.run(gan.loss.d_real, {gan.inputs.x: [[__x,__y] for __x, __y in zip(_x_sample, _y_sample)]})
                d.append(_d)
        contour = go.Contour(
            z = np.reshape(d, [-1]),
            x = np.reshape(x, [-1]),
            y = np.reshape(y, [-1]),
            opacity=0.5,
            showlegend=False,
            contours = dict(
                start=-0.5,
                end=0.5,
                size=0.03,
            )
        )
        print(np.shape(x), np.shape(y))
        #z = sess.run(gan.discriminator.sample, 

        global x_v, z_v
        if x_v is None:
            x_v = []
            z_v = []
            for j in range(args.sample_points // gan.batch_size()):
                _x_v, _z_v = sess.run([gan.inputs.x, gan.latent.sample])
                x_v.append(_x_v)
                z_v.append( _z_v)
            x_v = np.reshape(x_v, [-1,gan.inputs.x.shape[1]])
            z_v = np.reshape(z_v, [-1,gan.latent.sample.shape[1]])

        sample = []
        for j in range(args.sample_points // gan.batch_size()):
            offset = j*gan.batch_size()
            endoffset = (j+1)*gan.batch_size()
            z_v_sample = z_v[offset:endoffset]
            x_v_sample = x_v[offset:endoffset]
            _sample = sess.run(generator, {gan.inputs.x: x_v_sample, gan.latent.sample: z_v_sample})
            sample.append(_sample)
        sample = np.reshape(sample, [-1, 2])
        points = go.Scatter(x=sample[:,0], y=sample[:,1],
                mode='markers',
                marker = dict(
                    size = 10,
                    color = 'rgba(0, 152, 0, .8)',
                    line = dict(
                       width = 2,
                       color = 'rgb(0, 0, 0)'
                    )),
                name='fake')

        xpoints = go.Scatter(x=x_v[:,0], y=x_v[:,1],
                mode='markers',
                marker = dict(
                    size = 10,
                    color = 'rgba(255, 182, 193, .9)',
                    line = dict(
                       width = 2,
                       color = 'rgb(0, 0, 0)'
                    )),
                name='real')

        layout = go.Layout(hovermode='closest',
                xaxis=dict(range=[-1.5,1.5]),
                yaxis=dict(range=[-1.5,1.5]),
                width=1920,
                showlegend=False,
                height=1080
        )
        fig = go.Figure([contour, xpoints, points], layout=layout)
        data = pio.to_image(fig, format='png')
        #pio.write_image(fig,"sample.png")
        img = Image.open(io.BytesIO(data))
        #img = Image.open("sample.png").convert("RGB")
        #img.save("save.jpg")
        #plt.savefig(filename)
        self.plot(np.array(img), filename, save_samples, regularize=False)
        return [{'image': filename, 'label': '2d'}]

config = lookup_config(args)
if args.action == 'search':
    config = hc.Config(json.loads(open(os.getcwd()+'/randomsearch.json', 'r').read()))
    config['trainer']['rbbr']['optimizer']['optimizer']['learn_rate'] = random.choice([0.1,0.01,0.001, 0.005, 0.0001])
    config['trainer']['rbbr']['optimizer']['optimizer']['beta1'] = random.choice([0.1, 0.0001, 0.5, 0.9, 0.999])
    config['trainer']['rbbr']['optimizer']['optimizer']['beta2'] = random.choice([0.1, 0.0001, 0.5, 0.9, 0.999])
    config['trainer']['rbbr']['optimizer']['beta'] = random.choice([0, 1, 0.5, 0.99, 0.1])
    config['trainer']['rbbr']['optimizer']['gamma'] = random.choice([0, 1, 0.5, 0.99, 0.1, 10])
    config['trainer']['rbbr']['optimizer']['rho'] = random.choice([0, 1, 0.5, 0.99, 0.1])

def train(config, args):
    title = "[hypergan] 2d-test " + args.config
    GlobalViewer.title = title
    GlobalViewer.enabled = args.viewer

    with tf.device(args.device):
        config.generator['end_features'] = 2
        config.generator["class"]="class:__main__.Custom2DGenerator" # TODO
        config.discriminator["class"]="class:__main__.Custom2DDiscriminator" # TODO
        gan = hg.GAN(config, inputs = Custom2DInputDistribution(args))
        gan.name = args.config

        accuracy_x_to_g=distribution_accuracy(gan.inputs.x, gan.generator.sample)
        accuracy_g_to_x=distribution_accuracy(gan.generator.sample, gan.inputs.x)

        sampler = Custom2DSampler(gan)
        gan.selected_sampler = sampler

        tf.train.start_queue_runners(sess=gan.session)
        samples = 0
        steps = args.steps
        sampler.sample("samples/000000.png", args.save_samples)

        metrics = [accuracy_x_to_g, accuracy_g_to_x]
        sum_metrics = [0 for metric in metrics]
        for i in range(steps):
            gan.step()

            if args.viewer and i % args.sample_every == 0:
                samples += 1
                print("Sampling "+str(samples), args.save_samples)
                sample_file="samples/%06d.png" % (samples)
                sampler.sample(sample_file, args.save_samples)

            if i > steps * 9.0/10:
                for k, metric in enumerate(gan.session.run(metrics)):
                    sum_metrics[k] += metric 
            if i % 300 == 0:
                for k, metric in enumerate(gan.metrics().keys()):
                    metric_value = gan.session.run(gan.metrics()[metric])
                    print("--", metric,  metric_value)
                    if math.isnan(metric_value) or math.isinf(metric_value):
                        print("Breaking due to invalid metric")
                        return None

        tf.reset_default_graph()
        gan.session.close()

    return sum_metrics

if args.action == 'train':
    metrics = train(config, args)
    print("Resulting metrics:", metrics)
elif args.action == 'search':
    metric_sum = train(config, args)
    if 'search_output' in args:
        search_output = args.search_output
    else:
        search_output = "2d-test-results.csv"

    config_filename = "2d-measure-accuracy-"+str(uuid.uuid4())+'.json'
    hc.Selector().save(config_filename, config)
    with open(search_output, "a") as myfile:
        total = sum(metric_sum)
        myfile.write(config_filename+","+",".join([str(x) for x in metric_sum])+","+str(total)+"\n")
else:
    print("Unknown action: "+args.action)

