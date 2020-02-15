from common import *
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.gans.base_gan import BaseGAN
from hypergan.viewer import GlobalViewer
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import copy
import glob
import hyperchamber as hc
import hypergan as hg
import numpy as np
import os
import random
import re
import time
import torch
import torch.nn.functional as F
import uuid

arg_parser = ArgumentParser("render next frame")
parser = arg_parser.add_image_arguments()
parser.add_argument('--frames', type=int, default=4, help='Number of frames to embed.')
parser.add_argument('--shuffle', type=bool, default=False, help='Randomize inputs.')
args = arg_parser.parse_args()

width, height, channels = parse_size(args.size)

config = hg.configuration.Configuration.load(args.config+".json")

inputs = hg.inputs.image_loader.ImageLoader(args.batch_size, [args.directory],
                  channels=channels,
                  crop=args.crop,
                  width=width,
                  height=height,
                  resize=True,
                  shuffle=False)


class NextFrameGAN(BaseGAN):
    """ 
    """
    def __init__(self, *args, **kwargs):
        self.frames = kwargs.pop('frames')
        BaseGAN.__init__(self, *args, **kwargs)

    def create(self):
        self.latent = self.create_component("latent")
        self.ez = self.create_component("ez")
        self.ec = self.create_component("ec", input=self.ez)
        self.generator = self.create_component("generator", input=self.ec)
        self.discriminator = self.create_component("discriminator")
        self.loss = self.create_component("loss")
        self.trainer = self.create_component("trainer")

    def forward_discriminator(self, frames, c1, gs, cs):
        self.last_x = frames[-1]
        x = torch.cat(frames, dim=1)
        self.x = x
        D = self.discriminator
        d_real = D(x, context={"c": c1})
        d_fakes = []
        rems = frames[1:]
        for g, c in zip(gs, cs):
            d_fakes.append(D(torch.cat((*rems, g), dim=1), context={"c": c}))
            rems = rems[1:] + [g]

        d_fake = sum(d_fakes)/len(d_fakes)

        return d_real, d_fake

    def forward_loss(self):
        current_inputs = [self.inputs.next() for i in range(self.frames)]

        nxt_frames, nxt_encoding, real_frames = self.forward_gen(current_inputs)
        d_real, d_fake = self.forward_discriminator(current_inputs, real_frames[-1], nxt_frames, nxt_encoding)
        self.c = real_frames[-1]
        self.g = nxt_frames[0]
        self.d_fake = d_fake
        return self.loss.forward(d_real, d_fake)

    def forward_gen(self, xs):
        EZ = self.ez
        EC = self.ec
        G = self.generator

        c = torch.zeros((self.batch_size(), self.ec.current_channels, self.ec.current_height, self.ec.current_width)).cuda()
        z = torch.zeros((self.batch_size(), self.ez.current_channels, self.ez.current_height, self.ez.current_width)).cuda()

        real_cs = []
        for frame in xs:
            real_cs.append(c)
            z = EZ(frame, context={"z":z})
            c = EC(z, context={"c":c})

        gs = []
        cs = []
        for gen in range(5):
            cs.append(c)
            g = G(c, context={"z":z})
            z = EZ(g, context={"z":z})
            c = EC(z, context={"c":c})
            gs.append(g)
        self.last_g = g
        return gs, cs, real_cs

    def channels(self):
        return super(NextFrameGAN, self).channels() * self.frames

    def g_parameters(self):
        for param in self.generator.parameters():
            yield param
        for param in self.ec.parameters():
            yield param
        for param in self.ez.parameters():
            yield param

    def d_parameters(self):
        return self.discriminator.parameters()

    def regularize_gradient_norm(self, calculate_loss):
        x = Variable(self.x, requires_grad=True).cuda()
        d1_logits = self.discriminator(x, context={"c":self.c})
        d2_logits = self.d_fake

        loss = calculate_loss(d1_logits, d2_logits)

        if loss == 0:
            return [None, None]

        d1_grads = torch_grad(outputs=loss, inputs=x, create_graph=True)
        d1_norm = [torch.norm(_d1_grads.reshape(-1).cuda(),p=2,dim=0) for _d1_grads in d1_grads]
        reg_d1 = [((_d1_norm**2).cuda()) for _d1_norm in d1_norm]
        reg_d1 = sum(reg_d1)

        return loss, reg_d1

class VideoFrameSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.EZ = self.gan.ez
        self.EC = self.gan.ec
        self.G = self.gan.generator
        self.seed()

    def seed(self):
        self.c = torch.zeros((self.gan.batch_size(), self.EC.current_channels, self.EC.current_height, self.EC.current_width)).cuda()
        self.z = torch.zeros((self.gan.batch_size(), self.EZ.current_channels, self.EZ.current_height, self.EZ.current_width)).cuda()
        self.g = self.gan.inputs.next()

        for i in range(4):
            self.z = self.EZ(self.g, context={"z":self.z})
            self.c = self.EC(self.z, context={"c":self.c})
            self.g = self.gan.inputs.next()
            self.i = 0

    def _sample(self):
        self.z = self.EZ(self.g, context={"z":self.z})
        self.c = self.EC(self.z, context={"c":self.c})
        self.g = self.G(self.c, context={"z":self.z})
        if self.i % 100 == 0:
            self.seed()
        self.i += 1
        time.sleep(0.05)
        return [('input', self.gan.inputs.next()), ('generator', self.g)]


class TrainingVideoFrameSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)

    def _sample(self):
        gan = self.gan
        return [('input', self.gan.last_x), ('generator', self.gan.last_g)]


save_file = "saves/"+args.config+"/next-frame.save"

def setup_gan(config, inputs, args):
    gan = NextFrameGAN(config, inputs=inputs, frames=args.frames)
    gan.load(save_file)

    config_name = args.config
    GlobalViewer.title = "[hypergan] next-frame " + config_name
    GlobalViewer.enabled = args.viewer
    GlobalViewer.viewer_size = args.zoom

    return gan

def train(config, inputs, args):
    gan = setup_gan(config, inputs, args)
    sampler = TrainingVideoFrameSampler(gan)
    gan.selected_sampler = ""
    samples = 0

    #metrics = [batch_accuracy(gan.inputs.x, gan.uniform_sample), batch_diversity(gan.uniform_sample)]
    #sum_metrics = [0 for metric in metrics]
    for i in range(args.steps):
        gan.step()

        if args.action == 'train' and i % args.save_every == 0 and i > 0:
            print("saving " + save_file)
            gan.save(save_file)

        if i % args.sample_every == 0:
            sample_file="samples/"+args.config+"/%06d.png" % (samples)
            os.makedirs(os.path.expanduser(os.path.dirname(sample_file)), exist_ok=True)
            samples += 1
            sampler.sample(sample_file, args.save_samples)

        #if i > args.steps * 9.0/10:
        #    for k, metric in enumerate(gan.session.run(metrics)):
        #        print("Metric "+str(k)+" "+str(metric))
        #        sum_metrics[k] += metric 

    return []#sum_metrics

def sample(config, inputs, args):
    gan = setup_gan(config, inputs, args)
    sampler = VideoFrameSampler(gan)
    gan.selected_sampler = ""
    samples = 0
    for i in range(args.steps):
        sample_file="samples/"+args.config+"/%06d.png" % (samples)
        os.makedirs(os.path.expanduser(os.path.dirname(sample_file)), exist_ok=True)
        samples += 1
        sampler.sample(sample_file, args.save_samples)

if args.action == 'train':
    metrics = train(config, inputs, args)
    print("Resulting metrics:", metrics)
elif args.action == 'sample':
    sample(config, inputs, args)
else:
    print("Unknown action: "+args.action)
