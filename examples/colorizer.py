import os
import uuid
import random
import hypergan as hg
import hyperchamber as hc
import numpy as np
from hypergan.viewer import GlobalViewer
from hypergan.samplers.base_sampler import BaseSampler
from hypergan.gans.standard_gan import StandardGAN
from hypergan.layer_shape import LayerShape
from common import *
import torch.nn as nn
import torch

class Sampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.latent = self.gan.latent.next().data.clone()
        self.x = torch.cat([torch.unsqueeze(self.gan.x[0],0).repeat(gan.batch_size()//2,1,1,1), torch.unsqueeze(self.gan.x[1],0).repeat(gan.batch_size()//2,1,1,1)], 0)
        self.bw = BW(gan,None,None,hc.Config({}),LayerShape(*self.x.shape[1:])).forward_grayscale(self.x).repeat(1,3,1,1)
        self.gan = gan

    def _sample(self):
        gan = self.gan
        gan.x = self.bw
        return [
                 ('generator', self.bw),
                 ('g2', gan.generator.forward(self.latent))
               ]

class WalkSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.latent = self.gan.latent.next().data.clone()
        #self.x = torch.unsqueeze(self.gan.x[0],0).repeat(gan.batch_size(),1,1,1)
        self.x = self.gan.x
        self.bw = BW(gan,None,None,hc.Config({}),LayerShape(*self.x.shape[1:])).forward_grayscale(self.x).repeat(1,3,1,1)
        self.gan = gan

        self.latent1 = self.gan.latent.next()
        self.latent2 = self.gan.latent.next()
        self.velocity = 15/24.0
        direction = self.gan.latent.next()
        self.origin = direction
        self.pos = self.latent1
        self.direction = direction / torch.norm(direction, p=2, dim=1, keepdim=True).expand_as(direction)
        self.hardtanh = nn.Hardtanh()
        self.ones = torch.ones_like(self.direction, device="cuda:0")
        self.xstep = 0
        self.xstep_count = 120

    def _sample(self):
        gan = self.gan
        gan.x = self.bw
        self.xstep+=1
        if self.xstep > self.xstep_count:
            self.x = gan.inputs.next()
            #self.x = torch.unsqueeze(self.gan.x[0],0).repeat(gan.batch_size(),1,1,1)
            self.bw = BW(gan,None,None,hc.Config({}),LayerShape(*self.x.shape[1:])).forward_grayscale(self.x).repeat(1,3,1,1)
            self.xstep = 0

        self.pos = self.direction * self.velocity + self.pos
        self.gan.latent.z = self.pos
        mask = torch.gt(self.pos, self.ones)
        mask += torch.lt(self.pos, -self.ones)
        self.direction = self.direction + 2 * self.direction * (-self.ones * mask)


        g = gan.generator.forward(self.pos)

        return [
                 ('generator', g)
               ]


class BW(nn.Module):
    def __init__(self, gan, net, args, options, current_size):
        super().__init__()
        self.gan = gan
        self.replace = options.replace or False
        self.downsize = options.downsize or False
        x = self.gan.discriminator_real_inputs()[0]
        if self.replace is False:
            s = current_size.dims
            self.upsample = nn.Upsample((s[1], s[2]), mode="bilinear")
            self.shape = [1, s[1], s[2]]
        else:
            self.shape = [1, x.shape[2], x.shape[3]]
            self.upsample = None
        if self.downsize:
            self.downsize = [int(x) for x in self.downsize.split("*")]
            self.downsample = nn.Upsample(self.downsize, mode="bilinear")

    def forward(self, input):
        x = self.gan.x
        if self.downsize:
            x = self.downsample(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return self.forward_grayscale(x)

    def forward_grayscale(self, x):
        return x.mean(axis=1, keepdims=True)

    def layer_shape(self, current_size):
        shape = self.shape
        return LayerShape(*shape)

arg_parser = ArgumentParser("Colorize an image")
arg_parser.add_image_arguments()
args = arg_parser.parse_args()
GlobalViewer.set_options(enabled = True, title = "[hypergan] colorizer " + args.config, viewer_size = 1)

width, height, channels = parse_size(args.size)

config = lookup_config(args)

input_config = hc.Config({
    "class": "class:hypergan.inputs.image_loader.ImageLoader",
    "batch_size": args.batch_size,
    "directories": args.directory,
    "channels": channels,
    "crop": args.crop,
    "height": height,
    "random_crop": False,
    "resize": True,
    "shuffle": True,
    "width": width
})
klass = GANComponent.lookup_function(None, input_config['class'])
inputs = klass(input_config)

config_name = args.config
save_file = "saves/"+config_name+"/model.ckpt"
os.makedirs(os.path.expanduser(os.path.dirname(save_file)), exist_ok=True)

def setup_gan(config, inputs, args):
    gan = hg.GAN(config, inputs=inputs)

    gan.load(save_file)

    return gan

def train(config, inputs, args):
    gan = setup_gan(config, inputs, args)
    gan.name = config_name
    gan.selected_sampler = ""
    sampler = Sampler(gan)
    samples = 0

    for i in range(args.steps):
        gan.step()

        if args.action == 'train' and i % args.save_every == 0 and i > 0:
            print("saving " + save_file)
            gan.save(save_file)

        if i % args.sample_every == 0:
            sample_file="samples/"+config_name+"/%06d.png" % (samples)
            os.makedirs(os.path.expanduser(os.path.dirname(sample_file)), exist_ok=True)
            samples += 1
            sampler.sample(sample_file, args.save_samples)

def sample(config, inputs, args):
    gan = setup_gan(config, inputs, args)
    sampler = gan.sampler_for("sampler", args.sampler or WalkSampler)(gan)
    samples = 0
    for i in range(args.steps):
        sample_file="samples/"+config_name+"/%06d.png" % (samples)
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
