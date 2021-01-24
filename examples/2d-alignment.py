from PIL import Image
from chart_studio import plotly
from common import *
from hypergan.generators import *
from hypergan.search.random_search import RandomSearch
from hypergan.viewer import GlobalViewer
from hypergan.samplers.base_sampler import BaseSampler
import argparse
import hyperchamber as hc
import hypergan as hg
import numpy as np
import io
import json
import math
import os
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.io as pio
import torch
import uuid


arg_parser = ArgumentParser("Test your gan vs a known distribution", require_directory=False)
arg_parser.parser.add_argument('--distribution', '-t1', type=str, default='circle', help='source distribution, options are horizontal, vertical, circle')
arg_parser.parser.add_argument('--distribution2', '-t2', type=str, default='circle', help='target distribution, options are horizontal, vertical, circle')
arg_parser.parser.add_argument('--contour_size', '-cs', type=int, default=128, help='number of points to plot the discriminator contour with.  must be a multiple of batch_size')
arg_parser.parser.add_argument('--sample_points', '-p', type=int, default=512, help='number of scatter points to plot.  must be a multiple of batch_size')
args = arg_parser.parse_args()

config_filename = args.config

class Custom2DInputDistribution:
    def __init__(self, config):
        self.config = hc.Config(config)
        self.current_input_size = 2
        self.current_channels = 2
        self.x = torch.Tensor(self.config.batch_size, 2).cuda()
        self.y = torch.Tensor(self.config.batch_size, 2).cuda()

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self

    def batch_size(self):
        return self.config.batch_size

    def circle(self, x):
        spherenet = x**2
        spherenet = torch.sum(spherenet, dim=1)
        lam = torch.sqrt(spherenet).view([self.config.batch_size, 1])
        return x/lam#tf.reshape(lam,[int(lam.get_shape()[0]), 1])

    def modes(self, x):
        return (x*2)/2.0#+tf.random_normal(shape, 0, 0.04)

    def sample(self):
        self.x = self.distribution(self.x, args.distribution)
        self.y = self.distribution(self.y, args.distribution2)
        return [self.x, self.y]
    def distribution(self, x, distribution):
        if distribution == 'circle':
            x.normal_()
            x = self.circle(x)
        elif distribution == 'modes':
            x.uniform_()
            x = self.modes(x*2-1.0)
        elif distribution == 'modal-gaussian':
            x.uniform_()
            self.y.normal_()
            x = torch.round(x*2.0-1) + self.y*0.04+0.15
        elif distribution == 'sin':
            x = torch.rand((1, args.batch_size)).cuda()*21-10.5
            x = torch.transpose(x)
            r_data = torch.randn((args.batch_size,1), mean=0, stddev=0.1).cuda()
            xy = torch.sin(0.75*x)*7.0+x*0.5+r_data*1.0
            x = torch.cat([xy,x], 1)/16.0
        elif distribution == 'static-point':
            x = torch.ones([2]).cuda()
        elif distribution == 'vertical':
            x.uniform_()
            x[:,0]=torch.zeros([1], device='cuda:0')
        elif distribution == 'horizontal':
            x.uniform_()
            x[:,1]=torch.zeros([1], device='cuda:0')

        return x

    def width(self):
        return 1

    def height(self):
        return 1

    def channels(self):
        return 2

    def next(self, index=0):
        return self.sample()[index]

x_v, z_v = None, None
class Custom2DSampler(BaseSampler):
    def __init__(self, gan):
        self.gan = gan

    def sample(self, filename, save_samples):
        gan = self.gan

        config = gan.config

        #contours = args.contour_size

        #x,y = np.meshgrid(np.arange(-1.5, 1.5, 3/contours), np.arange(-1.5, 1.5, 3/contours))
        #d = []
        #for i in range(args.contour_size):
        #    _x = np.reshape(x[:,i], [-1]) 
        #    _y = np.reshape(y[:,i], [-1]) 
        #    for j in range(args.contour_size // gan.batch_size()):
        #        offset = j*gan.batch_size()
        #        endoffset = (j+1)*gan.batch_size()
        #        _x_sample = _x[offset:endoffset]
        #        _y_sample = _y[offset:endoffset]
        #        _d = gan.discriminator(torch.Tensor([[__x,__y] for __x, __y in zip(_x_sample, _y_sample)]).cuda()).detach().cpu().numpy()
        #        d.append(_d)
        #contour = go.Contour(
        #    z = np.reshape(d, [-1]),
        #    x = np.reshape(x, [-1]),
        #    y = np.reshape(y, [-1]),
        #    opacity=0.5,
        #    showlegend=False,
        #    contours = dict(
        #        start=0.4,
        #        end=2.0,
        #        size=0.03,
        #    )
        #)
        #z = sess.run(gan.discriminator.sample, 

        global x_v, z_v, y_v
        if x_v is None:
            x_v = []
            y_v = []
            z_v = []
            for j in range(args.sample_points // gan.batch_size()):
                x_v.append(gan.inputs.next(0).detach().clone())
                y_v.append(gan.inputs.next(1).detach().clone())
                z_v.append(gan.latent.sample().detach().clone())

        samples = []
        for j in range(args.sample_points // gan.batch_size()):
            z_v_sample = z_v[j]
            x_v_sample = x_v[j]
            y_v_sample = y_v[j]
            if gan.config.use_latent:
                sample = gan.generator(z_v_sample)
            else:
                if gan.config.ali:
                    sample = gan.generator(gan.encoder(x_v_sample))
                else:
                    sample = gan.generator(x_v_sample)
            samples.append(sample)
        sample = torch.cat(samples, dim=0).detach().cpu().numpy()
        points = go.Scatter(x=sample[:,0], y=sample[:,1],
                mode='markers',
                marker = dict(
                    size = 10,
                    line = dict(
                       width = 2
                    )),
                name='fake')

        x_v_np = torch.cat(x_v, dim=0).detach().cpu().numpy()
        xpoints = go.Scatter(x=x_v_np[:,0], y=x_v_np[:,1],
                mode='markers',
                marker = dict(
                    size = 10,
                    line = dict(
                       width = 2
                    )),
                name='real')

        y_v_np = torch.cat(y_v, dim=0).detach().cpu().numpy()
        ypoints = go.Scatter(x=y_v_np[:,0], y=y_v_np[:,1],
                mode='markers',
                marker = dict(
                    size = 10,
                    line = dict(
                       width = 2
                    )),
                name='real')


        layout = go.Layout(hovermode='closest',
                xaxis=dict(range=[-1.5,1.5]),
                yaxis=dict(range=[-1.5,1.5]),
                width=960,
                showlegend=False,
                height=480
        )
        fig = go.Figure([ypoints, xpoints, points], layout=layout)
        data = pio.to_image(fig, format='png')
        #pio.write_image(fig,filename)
        img = Image.open(io.BytesIO(data))
        #img = Image.open("sample.png").convert("RGB")
        #img.save("save.jpg")
        #plt.savefig(filename)
        self.plot_image(np.array(img), filename, save_samples, regularize=False)
        return [{'image': filename, 'label': '2d'}]

config = lookup_config(args)
if args.action == 'search':
    config_filename = "2d-measure-accuracy-"+str(uuid.uuid4())+'.json'
    config = hc.Config(json.loads(open(os.getcwd()+'/randomsearch.json', 'r').read()))
    config.trainer["optimizer"] = random.choice([{
        "class": "class:hypergan.optimizers.adamirror.Adamirror",
        "lr": random.choice(list(np.linspace(0.0001, 0.002, num=1000))),
        "betas":[random.choice([0.1, 0.9, 0.9074537537537538, 0.99, 0.999]),random.choice([0,0.9,0.997])]
    },{
        "class": "class:torch.optim.RMSprop",
        "lr": random.choice([1e-3, 1e-4, 5e-4, 3e-3]),
        "alpha": random.choice([0.9, 0.99, 0.999]),
        "eps": random.choice([1e-8, 1e-13]),
        "weight_decay": random.choice([0, 1e-2]),
        "momentum": random.choice([0, 0.1, 0.9]),
        "centered": random.choice([False, True])
    },
    {

        "class": "class:torch.optim.Adam",
        "lr": 1e-3,
        "betas":[random.choice([0.1, 0.9, 0.9074537537537538, 0.99, 0.999]),random.choice([0,0.9,0.997])],
        "eps": random.choice([1e-8, 1e-13]),
        "weight_decay": random.choice([0, 1e-2]),
        "amsgrad": random.choice([False, True])
        }

    ])

    config.trainer["hooks"].append(
      {
        "class": "function:hypergan.train_hooks.gradient_norm_train_hook.GradientNormTrainHook",
        "gamma": random.choice([1, 10, 1e-1, 100]),
        "loss": ["d"]
      })

    config.trainer["hooks"].append(
    {
      "class": "function:hypergan.train_hooks.online_ewc_train_hook.OnlineEWCTrainHook",
      "gamma": random.choice([0.5, 0.1, 0.9, 0.7]),
      "mean_decay": random.choice([0.9, 0.5, 0.99, 0.999, 0.1]),
      "skip_after_steps": random.choice([2000, 1000, 500]),
      "beta": random.choice([1e3, 1e4, 1e5, 1e2])
    })

    if(random.choice([False, True])):
        config.trainer["hooks"].append(
          {

            "class": "function:hypergan.train_hooks.extragradient_train_hook.ExtragradientTrainHook",
            "formulation": "agree"
          }
        )


def train(config, args):
    title = "[hypergan] 2d-test " + config_filename
    GlobalViewer.set_options(enabled = args.viewer, title = title, viewer_size=1)
    print("ARGS", args)

    gan = hg.GAN(config, inputs = Custom2DInputDistribution({
        "batch_size": args.batch_size
        }))
    trainable_gan = hg.TrainableGAN(gan, devices = args.devices, backend_name = args.backend)
    gan.name = config_filename
    if gan.config.use_latent:
        accuracy_x_to_g=lambda: distribution_accuracy(gan.inputs.next(1), gan.generator(gan.latent.next()))
        accuracy_g_to_x=lambda: distribution_accuracy(gan.generator(gan.latent.next()), gan.inputs.next(1))
    else:
        if gan.config.ali:
            accuracy_x_to_g=lambda: distribution_accuracy(gan.inputs.next(1), gan.generator(gan.encoder(gan.inputs.next())))
            accuracy_g_to_x=lambda: distribution_accuracy(gan.generator(gan.encoder(gan.inputs.next())), gan.inputs.next(1))
        else:
            accuracy_x_to_g=lambda: distribution_accuracy(gan.inputs.next(1), gan.generator(gan.inputs.next()))
            accuracy_g_to_x=lambda: distribution_accuracy(gan.generator(gan.inputs.next()), gan.inputs.next(1))

    sampler = Custom2DSampler(gan)
    gan.selected_sampler = sampler

    samples = 0
    steps = args.steps
    sample_file = "samples/"+config_filename+"/000000.png"
    os.makedirs(os.path.expanduser(os.path.dirname(sample_file)), exist_ok=True)
    sampler.sample(sample_file, args.save_samples)

    metrics = [accuracy_x_to_g, accuracy_g_to_x]
    sum_metrics = [0 for metric in metrics]
    broken = False
    for i in range(steps):
        if broken:
            break
        trainable_gan.step()

        if args.viewer and i % args.sample_every == 0:
            samples += 1
            print("Sampling "+str(samples))
            sample_file="samples/"+config_filename+"/%06d.png" % (samples)
            sampler.sample(sample_file, args.save_samples)

        if i % 100 == 0:
            for k, metric in enumerate(metrics):
                _metric =  metric().cpu().detach().numpy()
                sum_metrics[k] += _metric
                if not np.isfinite(_metric):
                    broken = True
                    break


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

    hc.Selector().save(config_filename, config)
    with open(search_output, "a") as myfile:
        total = sum(metric_sum)
        myfile.write(config_filename+","+",".join([str(x) for x in metric_sum])+","+str(total)+"\n")
else:
    print("Unknown action: "+args.action)

GlobalViewer.close()
