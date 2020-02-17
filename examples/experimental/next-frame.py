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
import torch.nn as nn
import torch.nn.functional as F
import uuid

arg_parser = ArgumentParser("render next frame")
parser = arg_parser.add_image_arguments()
parser.add_argument('--frames', type=int, default=4, help='Number of frames to embed.')
parser.add_argument('--shuffle', type=bool, default=False, help='Randomize inputs.')
args = arg_parser.parse_args()

GlobalViewer.set_options(enabled = args.viewer, title = "[hypergan] next-frame " + args.config, viewer_size = args.zoom)
width, height, channels = parse_size(args.size)

inputs = hg.inputs.image_loader.ImageLoader(args.batch_size, [args.directory],
                  channels=channels,
                  crop=args.crop,
                  width=width,
                  height=height,
                  resize=args.resize,
                  shuffle=False)

rand_inputs = hg.inputs.image_loader.ImageLoader(args.batch_size, [args.directory],
                  channels=channels,
                  crop=args.crop,
                  width=width,
                  height=height,
                  resize=args.resize,
                  shuffle=True)


class NextFrameGAN(BaseGAN):
    """ 
    """
    def __init__(self, *args, **kwargs):
        self.frames = kwargs.pop('frames')
        self.rand_inputs = kwargs.pop('rand_inputs')
        BaseGAN.__init__(self, *args, **kwargs)

    def create(self):
        self.latent = self.create_component("latent")
        self.ez = self.create_component("ez")
        self.ec = self.create_component("ec", input=self.ez)
        self.generator = self.create_component("generator", input=self.ec)
        if self.config.discriminator:
            self.discriminator = self.create_component("discriminator")
        if self.config.video_discriminator:
            self.video_discriminator = self.create_component("video_discriminator")
        if self.config.image_discriminator:
            self.image_discriminator = self.create_component("image_discriminator")
        if self.config.c_discriminator:
            self.c_discriminator = self.create_component("c_discriminator")
        self.loss = self.create_component("loss")
        self.trainer = self.create_component("trainer")

    def forward_discriminator(self, frames, x, cs, gs, gcs, rgs, rcs):
        d_fakes = []

        self.x = x
        D = self.discriminator
        d_real = D(x, context={"c": cs[-1]})
        rems = frames[1:]
        for g, c in zip(gs, gcs):
            #if config.discriminator3d:
            #    g = g[:, :, None, :, :]
            d_fakes.append(D(torch.cat((*rems, g), dim=1), context={"c": c}))
            rems = rems[1:] + [g]

        #if self.config.random:
        #    d_fakes.append(D(torch.cat(rgs, dim=1)))

        d_fake = sum(d_fakes)/len(d_fakes)
        return d_real, d_fake

    def forward_video_discriminator(self, cs, gcs, rcs):
        d_fakes = []
        VD = self.video_discriminator
        self.cs_cat = torch.cat(cs, dim=1)
        d_real = VD(self.cs_cat)
        rems = cs[1:]
        for c in gcs:
            d_fakes.append(VD(torch.cat((*rems, c), dim=1)))
            rems = rems[1:] + [c]

        if self.config.random:
            randcs = rcs[:len(rems)]
            for c in rcs[len(rems):]:
                d_fakes.append(VD(torch.cat((*randcs, c), dim=1)))
                randcs = randcs[1:] + [c]

        d_fake = sum(d_fakes)/len(d_fakes)
        return d_real, d_fake

    def forward_image_discriminator(self, x, gs, rgs):
        d_fakes = []
        ID = self.image_discriminator
        d_real = ID(x)
        igs = gs

        if self.config.random:
            igs += rgs
        for g in igs:
            d_fakes.append(ID(g))

        d_fake = sum(d_fakes)/len(d_fakes)
        return d_real, d_fake

    def forward_c_discriminator(self, c, gcs):
        d_fakes = []
        C = self.c_discriminator
        d_real = C(c)

        for g in gcs:
            d_fakes.append(C(g))

        d_fake = sum(d_fakes)/len(d_fakes)
        return d_real, d_fake

    def forward_loss(self):
        current_inputs = [self.inputs.next() for i in range(self.frames)]
        self.last_x = current_inputs[-1]

        cs, gs, gcs, rgs, rcs = self.forward_gen(current_inputs)
        self.xs, self.cs, self.gs, self.gcs, self.rgs, self.rcs = current_inputs, cs, gs, gcs, rgs, rcs

        d_loss = torch.tensor([0.0]).cuda()
        g_loss = torch.tensor([0.0]).cuda()

        if self.config.mse:
            mse = nn.MSELoss()
            next_inputs = [self.inputs.next() for i in range(self.frames)]
            mse_loss = mse(torch.cat(gs[:len(next_inputs)], dim=1), torch.cat(next_inputs, dim=1))
            d_loss += mse_loss
            g_loss += mse_loss
            self.add_metric("mse", mse_loss)

        if self.config.discriminator:
            self.x = torch.cat(current_inputs, dim=1)
            self.c = cs[-1]
            d_real, d_fake = self.forward_discriminator(current_inputs, self.x, cs, gs, gcs, rgs, rcs)
            self.od_fake = d_fake
            _d_loss, _g_loss = self.loss.forward(d_real, d_fake)
            d_loss += _d_loss
            g_loss += _g_loss

        if self.config.image_discriminator:
            self.ix = self.rand_inputs.next()
            d_real, d_fake = self.forward_image_discriminator(self.ix, gs, rgs)
            self.id_fake = d_fake
            _d_loss, _g_loss = self.loss.forward(d_real, d_fake)
            self.add_metric("ig_loss", _g_loss)
            self.add_metric("id_loss", _d_loss)
            d_loss += _d_loss
            g_loss += _g_loss

        if self.config.c_discriminator:
            self.c = cs[-1]
            d_real, d_fake = self.forward_c_discriminator(self.c, cs+gcs+rcs)
            self.cd_fake = d_fake
            _d_loss, _g_loss = self.loss.forward(d_real, d_fake)
            self.add_metric("cg_loss", _g_loss)
            self.add_metric("cd_loss", _d_loss)
            d_loss += _d_loss
            g_loss += _g_loss

        if self.config.video_discriminator:
            d_real, d_fake = self.forward_video_discriminator(cs, gcs, rcs)
            self.vd_fake = d_fake
            _d_loss, _g_loss = self.loss.forward(d_real, d_fake)
            self.add_metric("vg_loss", _g_loss)
            self.add_metric("vd_loss", _d_loss)
            d_loss += _d_loss
            g_loss += _g_loss

        return d_loss, g_loss

    def forward_gen(self, xs):
        EZ = self.ez
        EC = self.ec
        G = self.generator
        rgs = []
        rcs = []

        c = self.gen_c()
        z = self.gen_z()

        if(self.config.random):
            rg = G(c, context={"z":z})
            rgs.append(rg)

        cs = []
        for frame in xs:
            z = EZ(frame, context={"z":z})
            c = EC(z, context={"c":c})
            cs.append(c)
            if(self.config.random):
                rz = EZ(rg, context={"z":z})
                rc = EC(rz, context={"c":c})
                rg = G(c, context={"z":z})
                rgs.append(rg)
                rcs.append(rc)

        gs = []
        gcs = []
        for gen in range(self.config.forward_frames or 4):
            g = G(c, context={"z":z})
            z = EZ(g, context={"z":z})
            c = EC(z, context={"c":c})
            gcs.append(c)
            gs.append(g)
        self.last_g = g
        return cs, gs, gcs, rgs, rcs

    def channels(self):
        #if self.config.discriminator3d:
        #    return super(NextFrameGAN, self).channels()
        return super(NextFrameGAN, self).channels() * self.frames

    def gen_c(self):
        shape = [self.batch_size(), self.ec.current_channels, self.ec.current_height, self.ec.current_width]
        if self.config.dist == "uniform":
            return torch.rand(shape).cuda() * 2.0 - 1.0
        if self.config.dist == "vae":
            return torch.abs(torch.randn(shape).cuda() + torch.rand(*shape).cuda() * torch.randn(*shape).cuda())
        return torch.randn(shape).cuda()

    def gen_z(self):
        shape = (self.batch_size(), self.ez.current_channels, self.ez.current_height, self.ez.current_width)
        if self.config.zdist == "zeros":
            return torch.zeros(shape).cuda()

        return torch.randn(shape).cuda()

    def g_parameters(self):
        for param in self.generator.parameters():
            yield param
        for param in self.ec.parameters():
            yield param
        for param in self.ez.parameters():
            yield param

    def d_parameters(self):
        if self.config.discriminator:
            for param in self.discriminator.parameters():
                yield param
        if self.config.c_discriminator:
            for param in self.c_discriminator.parameters():
                yield param
        if self.config.video_discriminator:
            for param in self.video_discriminator.parameters():
                yield param
        if self.config.image_discriminator:
            for param in self.image_discriminator.parameters():
                yield param

    def regularize_gradient_norm(self, calculate_loss):
        loss = torch.Tensor([0.0]).cuda()
        if self.config.discriminator:
            x = Variable(self.x, requires_grad=True).cuda()
            d1_logits = self.discriminator(x, context={"c":self.c})
            d2_logits = self.od_fake
            loss = calculate_loss(d1_logits, d2_logits)

        #if self.config.video_discriminator:
        #    cs = Variable(self.cs_cat, requires_grad=True).cuda()
        #    d1_logits = self.video_discriminator(cs)
        #    d2_logits = self.vd_fake
        #    v_loss = calculate_loss(d1_logits, d2_logits)

        if self.config.image_discriminator:
            ix = Variable(self.ix, requires_grad=True).cuda()
            d1_logits = self.image_discriminator(ix)
            d2_logits = self.id_fake
            i_loss = calculate_loss(d1_logits, d2_logits)

        #if self.config.c_discriminator:
        #    c = Variable(self.gen_c(), requires_grad=True).cuda()
        #    cd1_logits = c
        #    cd2_logits = self.cd_fake
        #    c_loss = calculate_loss(cd1_logits, cd2_logits)

        d1_grads = []
        d1_norm = []
        if self.config.discriminator:
            d1_grads = torch_grad(outputs=loss, inputs=x, create_graph=True)
            d1_norm += [torch.norm(_d1_grads.reshape(-1).cuda(),p=2,dim=0) for _d1_grads in d1_grads]
        #$if self.config.video_discriminator:
        #$    d1_grads = torch_grad(outputs=v_loss, inputs=cs, create_graph=True)
        #$    d1_norm += [torch.norm(_d1_grads.reshape(-1).cuda(),p=2,dim=0) for _d1_grads in d1_grads]
        #$    loss += v_loss
        if self.config.image_discriminator:
            d1_grads = torch_grad(outputs=i_loss, inputs=ix, create_graph=True)
            d1_norm += [torch.norm(_d1_grads.reshape(-1).cuda(),p=2,dim=0) for _d1_grads in d1_grads]
            loss += i_loss
        #if self.config.c_discriminator:
        #    d1_c_grads = torch_grad(outputs=c_loss, inputs=c, create_graph=True)
        #    d1_norm += [torch.norm(_d1_grads.reshape(-1).cuda(),p=2,dim=0) for _d1_grads in d1_c_grads]
        #    loss += c_loss
        reg_d1 = [((_d1_norm**2).cuda()) for _d1_norm in d1_norm]
        reg_d1 = sum(reg_d1)

        return loss, reg_d1

class VideoFrameSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.EZ = self.gan.ez
        self.EC = self.gan.ec
        self.G = self.gan.generator
        self.inp = self.gan.inputs.next()
        self.seed()

    def seed(self):
        self.c = self.gan.gen_c()
        self.z = self.gan.gen_z()
        self.g = self.inp
        if self.gan.config.random:
            self.rg = self.G(self.c, context={"z":self.z})
            self.rz = self.z
            self.rc = self.c

        for i in range(3):
            self.z = self.EZ(self.g, context={"z":self.z})
            self.c = self.EC(self.z, context={"c":self.c})
            self.g = self.gan.inputs.next()
            self.i = 0

    def _sample(self):
        self.inp = self.gan.inputs.next()
        samples = [('input', self.inp)]
        self.z = self.EZ(self.g, context={"z":self.z})
        self.c = self.EC(self.z, context={"c":self.c})
        self.g = self.G(self.c, context={"z":self.z})
        if self.i % 100 == 0:
            self.seed()
        if self.gan.config.random:
            samples += [('rand', self.rg)]
            self.rz = self.EZ(self.rg, context={"z":self.rz})
            self.rc = self.EC(self.rz, context={"c":self.rc})
            self.rg = self.G(self.rc, context={"z":self.rz})
        self.i += 1
        time.sleep(0.033)
        samples += [('generator', self.g)]
        #time.sleep(0.1)
        return samples


class TrainingVideoFrameSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)

    def _sample(self):
        gan = self.gan
        return [('input', self.gan.last_x), ('generator', self.gan.last_g)]


save_file = "saves/"+args.config+"/next-frame.save"

def setup_gan(config, inputs, rand_inputs, args):
    gan = NextFrameGAN(config, inputs=inputs, rand_inputs=rand_inputs, frames=args.frames)
    gan.load(save_file)

    config_name = args.config

    return gan

def train(config, inputs, rand_inputs, args):
    gan = setup_gan(config, inputs, rand_inputs, args)
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
    gan = setup_gan(config, inputs, None, args)
    sampler = VideoFrameSampler(gan)
    gan.selected_sampler = ""
    samples = 0
    for i in range(args.steps):
        sample_file="samples/"+args.config+"/%06d.png" % (samples)
        os.makedirs(os.path.expanduser(os.path.dirname(sample_file)), exist_ok=True)
        samples += 1
        sampler.sample(sample_file, args.save_samples)

config = hg.configuration.Configuration.load(args.config+".json")

if args.action == 'train':
    metrics = train(config, inputs, rand_inputs, args)
    print("Resulting metrics:", metrics)
elif args.action == 'sample':
    sample(config, inputs, args)
else:
    print("Unknown action: "+args.action)
