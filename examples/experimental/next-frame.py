from hypergan.gan_component import ValidationException, GANComponent
from hypergan.gans.base_gan import BaseGAN
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from common import *
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
import torch.utils.data as data
import torchvision
import uuid

arg_parser = ArgumentParser("render next frame")
parser = arg_parser.add_image_arguments()
parser.add_argument('--frames', type=int, default=4, help='Number of frames to embed.')
args = arg_parser.parse_args()

width, height, channels = [int(x) for x in args.size.split('x')]

input_config = hc.Config({
    "batch_size": args.batch_size,
    "directories": [args.directory],
    "channels": channels,
    "crop": args.crop,
    "height": height,
    "random_crop": False,
    "resize": args.resize,
    "shuffle": args.action == "train",
    "width": width
})


class GreedyVideoFolder(torchvision.datasets.vision.VisionDataset):
    def __init__(self, root, frames, transform=None,
                 target_transform=None, is_valid_file=None):
        extensions = torchvision.datasets.folder.IMG_EXTENSIONS
        loader = torchvision.datasets.folder.default_loader
        super(GreedyVideoFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        samples = self._make_dataset(self.root, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.frames = frames
        self.loader = loader
        self.extensions = extensions

        self.samples = samples


    def _make_dataset(self, dir, extensions=None, is_valid_file=None):
        images = []
        dir = os.path.expanduser(dir)
        if not ((extensions is None) ^ (is_valid_file is None)):
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x):
                return torchvision.datasets.folder.has_file_allowed_extension(x, extensions)
        d = dir
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    images.append(path)

        return images

    def __getitem__(self, index):
        samples = []
        for sample in self.samples[index:index+self.frames]:
            path = sample
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            samples.append(sample)

        return [torch.cat(samples, dim=0)]

    def __len__(self):
        return len(self.samples) - self.frames


class GreedyVideoLoader:
    def __init__(self, frames, config):
        self.config = config
        self.datasets = []
        transform_list = []
        h, w = self.config.height, self.config.width

        if config.crop:
            transform_list.append(torchvision.transforms.CenterCrop((h, w)))

        if config.resize:
            transform_list.append(torchvision.transforms.Resize((h, w)))

        if config.random_crop:
            transform_list.append(torchvision.transforms.RandomCrop((h, w), pad_if_needed=True, padding_mode='edge'))

        transform_list.append(torchvision.transforms.ToTensor())
        transform = torchvision.transforms.Compose(transform_list)

        directories = self.config.directories or [self.config.directory]
        if(not isinstance(directories, list)):
            directories = [directories]

        self.dataloaders = []
        for directory in directories:
            #TODO channels
            image_folder = GreedyVideoFolder(directory, frames, transform=transform)
            self.dataloaders.append(data.DataLoader(image_folder, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=4, drop_last=True))
            self.datasets.append(iter(self.dataloaders[-1]))

    def batch_size(self):
        return self.config.batch_size

    def width(self):
        return self.config.width

    def height(self):
        return self.config.height

    def channels(self):
        return self.config.channels

    def next(self, index=0):
        try:
            self.sample = self.datasets[index].next()[0].cuda() * 2.0 - 1.0
            return self.sample
        except StopIteration:
            self.datasets[index] = iter(self.dataloaders[index])
            return self.next(index)

inputs = GreedyVideoLoader(args.frames, input_config)

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
        if self.config.generator_next:
            self.generator_next = self.create_component("generator_next", input=self.ec)
        if self.config.random_c:
            self.random_c = self.create_component("random_c", input=self.latent)
        if self.config.random_z:
            self.random_z = self.create_component("random_z", input=self.latent)
        if self.config.discriminator:
            self.discriminator = self.create_component("discriminator")
        if self.config.video_discriminator:
            self.video_discriminator = self.create_component("video_discriminator", input=self.ec)
        if self.config.image_discriminator:
            self.image_discriminator = self.create_component("image_discriminator", input=self.generator)
        if self.config.c_discriminator:
            self.c_discriminator = self.create_component("c_discriminator", input=self.ec)
        if self.config.predict_c:
            self.predict_c = self.create_component("predict_c", input=self.ec)
        if self.config.forward_c:
            self.forward_c = self.create_component("forward_c", input=self.ec)
        if self.config.nc:
            self.nc = self.create_component("nc", input=self.ez)
        if self.config.nz:
            self.nz = self.create_component("nz", input=self.ec)
        self.loss = self.create_component("loss")
        self.trainer = self.create_component("trainer")

    def forward_pass(self, frames, x, cs, gs, gcs, rgs, rcs):
        d_fakes = []

        D = self.discriminator
        if self.config.discriminator3d:
            if self.config.gcsf:
                c = gcs[0][:,:,None,:,:]
            else:
                c = cs[-1][:,:,None,:,:]
        else:
            c = cs[-1]

        d_real = D(x, context={"c": c})
        self.c = c
        if self.config.form == 3 or self.config.form == 2 or self.config.form == 5:
            if config.discriminator3d:
                grems = [x[:,:,None,:,:] for x in gs]
            else:
                grems = gs
            for i in range(len(grems) - self.frames + 1):
                rems = grems[i:i+self.frames]
                if config.discriminator3d:
                    d_fakes.append(D(torch.cat(rems, dim=2)))
                else:
                    d_fakes.append(D(torch.cat(rems, dim=1)))
        else:
            if config.discriminator3d:
                rems = [x[:,:,None,:,:] for x in frames]
            else:
                rems = frames
            for g, c in zip(gs, gcs):
                if config.discriminator3d:
                    g = g[:, :, None, :, :]
                    c = c[:, :, None, :, :]
                    rems = rems[1:] + [g]
                    d_fakes.append(D(torch.cat(rems, dim=2), context={"c": c}))
                else:
                    rems = rems[1:] + [g]
                    d_fakes.append(D(torch.cat(rems, dim=1), context={"c": c}))


        if len(rgs) > 0:
            grems = rgs[:len(rems)]
            rc = rcs[len(rems)-1]
            if config.discriminator3d:
                grems = [g[:,:,None,:,:] for g in grems]
                rc = rc[:,:,None,:,:]
                d_fakes.append(D(torch.cat(grems, dim=2), context={"c":rc}))
            else:
                d_fakes.append(D(torch.cat(grems, dim=1), context={"c":rc}))
            for rg, rc in zip(rgs[len(rems):], rcs[len(rems):]):
                if config.discriminator3d:
                    grems = grems[1:] + [rg[:,:,None,:,:]]
                    rc = rc[:,:,None,:,:]
                    d_fakes.append(D(torch.cat(grems, dim=2), context={"c":rc}))
                else:
                    grems = grems[1:] + [rg]
                    d_fakes.append(D(torch.cat(grems, dim=1), context={"c":rc}))

        d_fake = sum(d_fakes)/len(d_fakes)
        return d_real, d_fake

    def forward_video_discriminator(self, cs, gcs, rcs):
        d_fakes = []
        VD = self.video_discriminator
        if self.config.discriminator3d:
            cs = [c[:,:,None,:,:] for c in cs]
            self.cs_cat = torch.cat(cs, dim=2)
        else:
            self.cs_cat = torch.cat(cs, dim=1)
        d_real = VD(self.cs_cat)
        rems = cs
        for c in gcs:
            if self.config.discriminator3d:
                c = c[:,:,None,:,:]
            rems = rems[1:] + [c]
            if self.config.discriminator3d:
                next_cs = torch.cat(rems, dim=2)
            else:
                next_cs = torch.cat(rems, dim=1)

            d_fakes.append(VD(next_cs))

        if self.config.random:
            randcs = rcs[:len(rems)]
            for c in rcs[len(rems):]:
                if self.config.discriminator3d:
                    d_fakes.append(VD(torch.cat([b[:,:,None,:,:] for b in (*randcs[1:], c)], dim=2)))
                else:
                    d_fakes.append(VD(torch.cat((*randcs, c), dim=1)))
                randcs = randcs[1:] + [c]

        d_fake = sum(d_fakes)/len(d_fakes)
        return d_real, d_fake

    def forward_image_discriminator(self, x, gs):
        d_fakes = []
        ID = self.image_discriminator
        d_real = ID(x)

        for g in gs:
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
        current_inputs = list(torch.chunk(self.inputs.next(), self.frames, dim=1))
        self.last_x = current_inputs[-1]

        cs, zs, gs, gcs, gzs, rgs, rcs, vae_loss = self.forward_gen(current_inputs)
        self.xs, self.cs, self.gs, self.gcs, self.rgs, self.rcs = current_inputs, cs, gs, gcs, rgs, rcs
        self.zs, self.gzs = zs, gzs

        d_loss = torch.tensor([0.0]).cuda()
        g_loss = torch.tensor([0.0]).cuda()

        if self.config.vae:
            g_loss += vae_loss
            self.add_metric('vae', vae_loss)

        if self.config.mse:
            mse = nn.MSELoss()
            mse_loss = mse(self.mse_g, self.last_x)
            d_loss += mse_loss
            g_loss += mse_loss
            self.add_metric("mse", mse_loss)

        if self.config.regularize_c:
            all_cs = cs + gcs
            c_mean = sum([c.mean() for c in all_cs])/len(all_cs)
            self.add_metric("c_mean", c_mean)
            g_loss += c_mean

        if self.config.discriminator:
            if self.config.discriminator3d:
                self.x = torch.cat([x[:, :, None, :, :] for x in current_inputs], dim=2)
            else:
                self.x = torch.cat(current_inputs, dim=1)
            d_real, d_fake = self.forward_pass(current_inputs, self.x, cs, gs, gcs, rgs, rcs)
            self.od_fake = d_fake
            _d_loss, _g_loss = self.loss.forward(d_real, d_fake)
            d_loss += _d_loss
            g_loss += _g_loss

        if self.config.image_discriminator:
            self.ix = self.last_x

            d_real, d_fake = self.forward_image_discriminator(self.ix, gs)
            self.id_fake = d_fake
            _d_loss, _g_loss = self.loss.forward(d_real, d_fake)
            self.add_metric("ig_loss", _g_loss)
            self.add_metric("id_loss", _d_loss)
            d_loss += _d_loss
            g_loss += _g_loss

        if self.config.c_discriminator:
            self.c_real = self.gen_c()
            d_real, d_fake = self.forward_c_discriminator(self.c_real, [cs[self.frames-1]])
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

        if self.config.predict_c:
            g_loss += self.predict_c_loss
            self.add_metric("pc", self.predict_c_loss)

        return d_loss, g_loss

    def vae_loss(self, component):
        if hasattr(component, 'vae'):
            logvar = component.vae.sigma
            mu = component.vae.mu
            return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return 0.0


    def forward_gen(self, xs):
        EZ = self.ez
        EC = self.ec
        G = self.generator
        rgs = []
        rcs = []
        vae_loss = []

        c = self.gen_c()
        z = self.gen_z()
        first_c = c
        #c = self.random_c(self.latent.sample)
        #z = self.random_z(self.latent.sample)
        #z = self.gen_z()

        if(self.config.random):
            rc = self.random_c(self.latent.sample)
            if self.config.vae:
                vae_loss.append(self.vae_loss(self.random_c))

            #rc = self.gen_c()
            rz = self.random_z(self.latent.sample)
            rg = G(rc, context={"z":rz})
            rgs.append(rg)
            rcs.append(rc)
            gen_frames = self.config.random_frames
            if self.config.extra_long:
                for every, multiple in self.config.extra_long:
                    if self.steps % every == 0:
                        print("Running extra long step " + str(multiple) + " frames")
                        gen_frames = multiple

            for i in range(gen_frames):
                rz = EZ(rg, context={"z":rz})
                rc = EC(rz, context={"c":rc})
                if self.config.vae:
                    vae_loss.append(self.vae_loss(EC))
                    vae_loss.append(self.vae_loss(EZ))

                rg = G(rc, context={"z":rz})
                rgs.append(rg)
                rcs.append(rc)

        cs = []
        zs = []
        gs = []
        c_prev = c
        for i, frame in enumerate(xs):
            if self.config.form == 1:
                c = EC(z, context={"c":c})
                z = EZ(frame, context={"z":z})
            elif self.config.form == 2:
                z = EZ(frame, context={"z":z})
                c_prev = c
                c = EC(z, context={"c":c})
                g = G(c, context={"z":z})
                if self.config.encode_g:
                    gs.append(g)
            elif self.config.form == 5:
                z = EZ(frame, context={"z":z})
                c = EC(z, context={"c":c})
                g = G(c, context={"z":z})
                gs.append(g)

                self.mse_g = g
            elif self.config.form == 3:
                z = EZ(frame, context={"z":z})
                c = EC(z, context={"c":c})
                g = G(c, context={"z":z})
                gs.append(g)
            else:
                z = EZ(frame, context={"z":z})
                c = EC(z, context={"c":c})
            zs.append(z)
            cs.append(c)
            if self.config.vae:
                vae_loss.append(self.vae_loss(EC))
                vae_loss.append(self.vae_loss(EZ))

        input_c = c
        input_z = z

        gcs = []
        gzs = []
        gen_frames = self.config.forward_frames or self.frames + 1


        if self.config.extra_long:
            for every, multiple in self.config.extra_long:
                if self.steps % every == 0:
                    print("Running extra long step " + str(multiple) + " frames")
                    gen_frames = multiple
        for gen in range(gen_frames):
            if self.config.form == 1:
                c = EC(z, context={"c":c})
                g = G(c, context={"z":z})
                z = EZ(g, context={"z":z})
            elif self.config.form == 2:
                #fc = self.forward_c(c, {"c": c_prev})
                fc = self.nc(z, {"c": c})
                c_prev = c
                g = G(fc, context={"z":z})
                z = EZ(g, context={"z":z})
                c = EC(z, context={"c":c})
            elif self.config.form == 5:
                c = self.nc(z, {"c": c})
                g = G(c, context={"z":z})
                z = EZ(g, context={"z":z})
                c = EC(z, context={"c":c})

            elif self.config.form == 3:
                g = self.generator_next(c, context={"z":z})
                z = EZ(g, context={"z":z})
                c = EC(z, context={"c":c})
            else:
                g = G(c, context={"z":z})
                z = EZ(g, context={"z":z})
                c = EC(z, context={"c":c})
            gcs.append(c)
            gzs.append(z)
            gs.append(g)
        if self.config.form != 1 and self.config.form != 3 and self.config.form != 2 and self.config.form != 5:
            g = G(c, context={"z":z})
            gs.append(g)
        self.last_g = g

        if self.config.predict_c:
            self.predict_c_loss = nn.MSELoss()(self.predict_c(first_c), first_c)
            for i, frame in enumerate(xs):
                c = self.predict_c(c)
                z = EZ(frame, context={"z":z})
                c = EC(z, context={"c":c})
                g = G(c, context={"z":z})
                gzs.append(z)
                gcs.append(c)
                gs.append(g)

        if self.config.vae:
            vae_loss = sum(vae_loss)/len(vae_loss)
        return cs, zs, gs, gcs, gzs, rgs, rcs, vae_loss

    def channels(self):
        if self.config.discriminator3d:
            return super(NextFrameGAN, self).channels()
        return super(NextFrameGAN, self).channels() * self.frames

    def gen_c(self):
        shape = [self.batch_size(), self.ec.current_channels, self.ec.current_height, self.ec.current_width]
        if self.config.cdist == "random":
            return self.random_c(self.latent.sample)
        if self.config.cdist == "uniform":
            return torch.rand(shape).cuda()
        if self.config.cdist == "uniform_1_to_1":
            return torch.rand(shape).cuda() * 2.0 - 1
        if self.config.cdist == "vae":
            return torch.abs(torch.randn(shape).cuda() + torch.rand(*shape).cuda() * torch.randn(*shape).cuda())
        if self.config.cdist == "zeros":
            return torch.zeros(shape).cuda()
        return torch.abs(torch.randn(shape).cuda())

    def gen_z(self):
        shape = (self.batch_size(), self.ez.current_channels, self.ez.current_height, self.ez.current_width)
        if self.config.zdist == "random":
            return self.random_z(self.latent.sample)
        if self.config.zdist == "uniform":
            return torch.rand(shape).cuda() * 2.0 - 1
        if self.config.zdist == "uniform_1_to_1":
            return torch.rand(shape).cuda() * 2.0 - 1
        if self.config.zdist == "zeros":
            return torch.zeros(shape).cuda()

        return torch.abs(torch.randn(shape).cuda())

    def g_parameters(self):
        for param in self.generator.parameters():
            yield param
        for param in self.ec.parameters():
            yield param
        for param in self.ez.parameters():
            yield param

        if self.config.generator_next:
            for param in self.generator_next.parameters():
                yield param
        if self.config.random_c:
            for param in self.random_c.parameters():
                yield param
        if self.config.random_z:
            for param in self.random_z.parameters():
                yield param
        if self.config.predict_c:
            for param in self.predict_c.parameters():
                yield param
        if self.config.forward_c:
            for param in self.forward_c.parameters():
                yield param
        if self.config.nz:
            for param in self.nz.parameters():
                yield param
        if self.config.nc:
            for param in self.nc.parameters():
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

class VideoFrameSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.EZ = self.gan.ez
        self.EC = self.gan.ec
        self.G = self.gan.generator
        self.refresh_input_cache()
        self.seed()

    def seed(self):
        self.c = self.gan.gen_c()
        self.z = self.gan.gen_z()
        #self.z = self.gan.random_z(self.gan.latent.sample)
        #self.c = self.gan.random_c(self.gan.latent.sample)
        if self.gan.config.random:
            self.rz = self.gan.random_z(self.gan.latent.sample)
            self.rc = self.gan.random_c(self.gan.latent.sample)
            #self.rc = self.c
            #self.rc = self.gan.gen_c()
            self.rg = self.G(self.rc, context={"z":self.rz})

        for i in range(self.gan.frames):
            self.g = self.input_cache[i]
            if self.gan.config.form == 1:
                self.c = self.EC(self.z, context={"c":self.c})
                self.z = self.EZ(self.g, context={"z":self.z})
            elif self.gan.config.form == 2:
                self.z = self.EZ(self.g, context={"z":self.z})
                self.c_prev = self.c
                self.c = self.EC(self.z, context={"c":self.c})
                if self.gan.config.encode_g:
                    self.g = self.G(self.c, context={"z":self.z})
            elif self.gan.config.form == 5:
                self.z = self.EZ(self.g, context={"z":self.z})
                self.c = self.EC(self.z, context={"c":self.c})
                self.g = self.G(self.c, context={"z":self.z})
            else:
                self.z = self.EZ(self.input_cache[i], context={"z":self.z})
                self.c = self.EC(self.z, context={"c":self.c})
            if self.gan.config.form == 3:
                self.g = self.gan.generator_next(self.c, context={"z":self.z})
            self.i = 0

    def refresh_input_cache(self):
        self.input_cache = list(torch.chunk(self.gan.inputs.next(), self.gan.frames, dim=1))
        for i in range(len(self.input_cache)-1):
            self.gan.inputs.next()
        self.input_idx = 0

    def next_input(self):
        if self.input_idx >= len(self.input_cache):
            self.refresh_input_cache()
        image = self.input_cache[self.input_idx]
        self.input_idx += 1
        return image

    def _sample(self):
        self.inp = self.next_input()
        samples = []
        samples += [('input', self.inp)]
        if self.gan.config.form == 2:
            #self.fc = self.gan.forward_c(self.c, context={"c":self.c_prev})
            self.fc = self.gan.nc(self.z, context={"c":self.c})
            self.c_prev = self.c
            self.g = self.G(self.fc, context={"z":self.z})
            self.z = self.EZ(self.g, context={"z":self.z})
            self.c = self.EC(self.z, context={"c":self.c})
        if self.gan.config.form == 5:
            self.c = self.gan.nc(self.z, context={"c":self.c})
            self.g = self.G(self.c, context={"z":self.z})
            self.z = self.EZ(self.g, context={"z":self.z})
            self.c = self.EC(self.z, context={"c":self.c})

        elif self.gan.config.form == 1:
            self.c = self.EC(self.z, context={"c":self.c})
            self.g = self.G(self.c, context={"z":self.z})
            self.z = self.EZ(self.g, context={"z":self.z})
        elif self.gan.config.form == 3:
            self.g = self.gan.generator_next(self.c, context={"z":self.z})
            self.z = self.EZ(self.g, context={"z":self.z})
            self.c = self.EC(self.z, context={"c":self.c})
        else:
            self.g = self.G(self.c, context={"z":self.z})
            self.z = self.EZ(self.g, context={"z":self.z})
            self.c = self.EC(self.z, context={"c":self.c})
        print(self.c.mean())
        if self.i % (4*24) == 0:
            print("RESET")
            self.seed()
        if self.gan.config.random:
            samples += [('rand', self.rg)]
            self.rz = self.EZ(self.rg, context={"z":self.rz})
            self.rc = self.EC(self.rz, context={"c":self.rc})
            self.rg = self.G(self.rc, context={"z":self.rz})
        self.i += 1
        samples += [('generator', self.g)]
        return samples


class TrainingVideoFrameSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)

    def _sample(self):
        gan = self.gan
        return [('input', self.gan.last_x), ('generator', self.gan.gs[-1])]


save_file = "saves/"+args.config+"/next-frame.save"

def setup_gan(config, inputs, args):
    gan = NextFrameGAN(config, inputs=inputs, frames=args.frames)
    gan.load(save_file)

    config_name = args.config

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

config = hg.configuration.Configuration.load(args.config+".json")

if args.action == 'train':
    metrics = train(config, inputs, args)
    print("Resulting metrics:", metrics)
elif args.action == 'sample':
    sample(config, inputs, args)
else:
    print("Unknown action: "+args.action)
