from hypergan.gan_component import ValidationException, GANComponent
from hypergan.inputs.crop_resize_transform import CropResizeTransform
from hypergan.gans.base_gan import BaseGAN
from hypergan.layer_shape import LayerShape
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
from natsort import natsorted
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import uuid
from common import *

arg_parser = ArgumentParser("render next frame")
parser = arg_parser.add_image_arguments()
parser.add_argument('--frames', type=int, default=4, help='Number of frames to embed.')
parser.add_argument('--per_sample_frames', type=int, default=3, help='Number of frames to use at once.  Ex: 3 becomes x1, x2, g3')
args = arg_parser.parse_args()

if __name__ == "__main__":
    GlobalViewer.set_options(enabled = args.viewer, title = "[hypergan] next-frame " + args.config, viewer_size = args.zoom)
    width, height, channels = [int(x) for x in args.size.split('x')]

    input_config = hc.Config({
        "batch_size": args.batch_size,
        "directories": [args.directory],
        "channels": channels,
        "crop": args.crop,
        "height": height,
        "random_crop": False,
        "resize": True,
        "shuffle": args.action == "train",
        "width": width
    })
    config_name = args.config
    save_file = "saves/"+config_name+"/model.ckpt"
    os.makedirs(os.path.expanduser(os.path.dirname(save_file)), exist_ok=True)

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
        for root, _, fnames in natsorted(os.walk(d, followlinks=True)):
            for fname in natsorted(fnames):
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

        if config.random_crop:
            transform_list.append(torchvision.transforms.RandomCrop((h, w), pad_if_needed=True, padding_mode='edge'))
        else:
            transform_list.append(CropResizeTransform((h, w)))

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

if __name__ == "__main__":
    inputs = GreedyVideoLoader(args.frames, input_config)

class NextFrameGAN(BaseGAN):
    """ 
    """
    def __init__(self, *args, **kwargs):
        self.per_sample_frames=3
        self.frames = kwargs.pop('frames')
        self.per_sample_frames = kwargs.pop('per_sample_frames')
        BaseGAN.__init__(self, *args, **kwargs)

    def create(self):
        self.latent = self.create_component("latent")
        z_w = 16
        z_h = 16
        z_channels = 256
        z_shape = LayerShape(z_channels, z_h, z_w)
        c_w = 8
        c_h = 8
        c_channels = 512
        c_shape = LayerShape(c_channels, c_h, c_w)
        self.ez = self.create_component("ez", context_shapes={"z": z_shape})
        self.ec = self.create_component("ec", input=self.ez, context_shapes={"c": c_shape})
        self.gz = self.create_component("gz", input=self.ec, context_shapes={"z": z_shape})
        self.generator = self.create_component("generator", input=self.ez)
        if self.config.generator_next:
            self.generator_next = self.create_component("generator_next", input=self.ec)
        if self.config.random_c:
            self.random_c = self.create_component("random_c", input=self.latent)
        if self.config.random_z:
            self.random_z = self.create_component("random_z", input=self.latent)
        if self.config.discriminator:
            self.discriminator = self.create_component("discriminator", context_shapes={"c": c_shape})
        if self.config.video_discriminator:
            self.video_discriminator = self.create_component("video_discriminator", input=self.ec)
        if self.config.image_discriminator:
            self.image_discriminator = self.create_component("image_discriminator", input=self.generator)
        if self.config.c_discriminator:
            self.c_discriminator = self.create_component("c_discriminator", input=self.ec)
        if self.config.nc:
            self.nc = self.create_component("nc", input=self.ez)
        if self.config.nz:
            self.nz = self.create_component("nz", input=self.ec)

    def forward_discriminator(self, inputs):
        return self.discriminator(inputs[0])

    def forward_pass(self, frames, xs, cs, gs, gcs, rgs, rcs, loss):
        d_fakes = []
        d_reals = []
        d_losses = []
        g_losses = []

        D = self.discriminator
        if self.config.discriminator3d:
            if self.config.gcsf:
                c = gcs[0][:,:,None,:,:]
            else:
                c = cs[-1][:,:,None,:,:]
        else:
            c = cs[-1]
        cx = self.gen_c()
        zx = self.gen_z()
        cg = self.gen_c()
        zg = self.gen_z()
        EZ = self.ez
        EC = self.ec
        GZ = self.gz

        self.d_fake_inputs = []
        d_real = D(xs[0])
        self.d_real = d_real

        rems = [None] + frames[1:self.per_sample_frames]
        for g in gs[self.per_sample_frames:]:
            rems = rems[1:] + [g]
            d_fake_input = torch.cat(rems, dim=1)
            self.d_fake_inputs.append(d_fake_input.clone().detach())
            d_fake = D(d_fake_input)
            _d_loss, _g_loss = loss.forward(d_real, d_fake)
            d_losses.append(_d_loss)
            g_losses.append(_g_loss)

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
                grems = grems[1:] + [rg]
                d_fakes.append(D(torch.cat(grems, dim=1), context={"c":rc}))

        d_loss = sum(d_losses)/len(d_losses)
        g_loss = sum(g_losses)/len(g_losses)
        self.d_fake = sum(d_fake)/len(d_fake)
        return d_loss, g_loss

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

    def forward_loss(self, loss=None):
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
                self.x = torch.cat(current_inputs[:self.per_sample_frames], dim=1)
                self.xs = []
                for i in range(self.frames-self.per_sample_frames+1):
                    self.xs += [torch.cat(current_inputs[i:i+self.per_sample_frames], dim=1)]
            _d_loss, _g_loss = self.forward_pass(current_inputs, self.xs, cs, gs, gcs, rgs, rcs, loss)
            g_loss += _g_loss
            d_loss += _d_loss

        if self.config.image_discriminator:
            self.ix = self.last_x

            d_real, d_fake = self.forward_image_discriminator(self.ix, gs)
            self.id_fake = d_fake
            _d_loss, _g_loss = loss.forward(d_real, d_fake)
            self.add_metric("ig_loss", _g_loss)
            self.add_metric("id_loss", _d_loss)
            d_loss += _d_loss
            g_loss += _g_loss

        if self.config.c_discriminator:
            self.c_real = self.gen_c()
            d_real, d_fake = self.forward_c_discriminator(self.c_real, [cs[self.frames-1]])
            self.cd_fake = d_fake
            _d_loss, _g_loss = loss.forward(d_real, d_fake)
            self.add_metric("cg_loss", _g_loss)
            self.add_metric("cd_loss", _d_loss)
            d_loss += _d_loss
            g_loss += _g_loss

        if self.config.video_discriminator:
            d_real, d_fake = self.forward_video_discriminator(cs, gcs, rcs)
            self.vd_fake = d_fake
            _d_loss, _g_loss = loss.forward(d_real, d_fake)
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

    def discriminator_components(self):
        components = [self.discriminator]
        if self.config.c_discriminator:
            components.append(self.c_discriminator)
        if self.config.video_discriminator:
            components.append(self.video_discriminator)
        if self.config.image_discriminator:
            components.append(self.image_discriminator)
        return components

    def generator_components(self):
        components = [self.generator, self.ec, self.ez, self.gz]
        if self.config.generator_next:
            components.append(self.generator_next)
        if self.config.random_c:
            components.append(self.random_c)
        if self.config.random_z:
            components.append(self.random_z)
        if self.config.nc:
            components.append(self.nc)
        if self.config.nz:
            components.append(self.nz)
        return components

    def forward_gen(self, xs):
        EZ = self.ez
        EC = self.ec
        G = self.generator
        GZ = self.gz
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
        for i, frame in enumerate(xs):
            z = EZ(frame, context={"z":z})
            c = EC(z, context={"c":c})
            g = G(z)
            zs.append(z)
            cs.append(c)
            gs.append(g)

        input_c = c
        input_z = z

        gcs = []
        gzs = []
        if self.config.forward_frames is None:
            gen_frames = self.frames + 1
        else:
            gen_frames = self.config.forward_frames


        if self.config.extra_long:
            for every, multiple in self.config.extra_long:
                if self.steps % every == 0:
                    print("Running extra long step " + str(multiple) + " frames")
                    gen_frames = multiple
        g = xs[-1]
        for gen in range(gen_frames):
            z2 = GZ(c, context={"z":z})
            g = G(z2)
            z = EZ(g, context={"z":z2})
            c = EC(z, context={"c":c})
            gcs.append(c)
            gzs.append(z2)
            gs.append(g)
        if self.config.vae:
            vae_loss.append(self.vae_loss(gcs[0]))
        self.last_g = g

        vae_loss = sum(vae_loss)
        return cs, zs, gs, gcs, gzs, rgs, rcs, vae_loss

    def channels(self):
        if self.config.discriminator3d:
            return super(NextFrameGAN, self).channels()
        return super(NextFrameGAN, self).channels() * self.per_sample_frames

    def gen_c(self):
        shape = [self.batch_size(), *self.ec.current_size.dims]
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
        shape = (self.batch_size(), self.ez.current_size.channels, self.ez.current_size.height, self.ez.current_size.width)
        if self.config.zdist == "random":
            return self.random_z(self.latent.sample)
        if self.config.zdist == "uniform":
            return torch.rand(shape).cuda() * 2.0 - 1
        if self.config.zdist == "uniform_1_to_1":
            return torch.rand(shape).cuda() * 2.0 - 1
        if self.config.zdist == "zeros":
            return torch.zeros(shape).cuda()

        return torch.abs(torch.randn(shape).cuda())

    def discriminator_fake_inputs(self):
        if hasattr(self, 'd_fake_inputs'):
            return list([[d_in] for d_in in self.d_fake_inputs]).copy()
        else:
            print("next gs")
            return [list(torch.chunk(self.inputs.next(), self.per_sample_frames, dim=1))[-1]]

    def discriminator_real_inputs(self):
        if hasattr(self, 'xs'):
            return self.xs
        else:
            print("next xs", self.per_sample_frames)
            return [list(torch.chunk(self.inputs.next(), self.per_sample_frames, dim=1))[-1]]

class VideoFrameSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.EZ = self.gan.ez
        self.EC = self.gan.ec
        self.G = self.gan.generator
        self.GZ = self.gan.gz
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
        self.g = self.input_cache[self.gan.frames-1]
        self.i=0
        for i in range(self.gan.frames):
            g = self.input_cache[i]
            self.z = self.EZ(g, context={"z":self.z})
            self.c = self.EC(self.z, context={"c":self.c})
            self.g = self.G(self.z)
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
        self.z = self.EZ(self.g, context={"z":self.z})
        self.c = self.EC(self.z, context={"c":self.c})
        self.z = self.GZ(self.c, context={"z":self.z})
        self.g = self.G(self.z)
        print(self.c.mean())
        if self.i % (8*24) == 0:
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
        return [('input', self.gan.last_x), ('gfirst', self.gan.gs[0]), ('generator', self.gan.gs[-1])]



def setup_gan(config, inputs, args):
    gan = NextFrameGAN(config, inputs=inputs, frames=args.frames, per_sample_frames=args.per_sample_frames)
    gan.load(save_file)

    config_name = args.config

    return gan

def train(config, inputs, args):
    gan = setup_gan(config, inputs, args)
    trainable_gan = hg.TrainableGAN(gan, save_file = save_file, devices = args.devices, backend_name = args.backend)
    sampler = TrainingVideoFrameSampler(gan)
    gan.selected_sampler = ""
    samples = 0

    #metrics = [batch_accuracy(gan.inputs.x, gan.uniform_sample), batch_diversity(gan.uniform_sample)]
    #sum_metrics = [0 for metric in metrics]
    for i in range(args.steps):
        trainable_gan.step()

        if args.action == 'train' and i % args.save_every == 0 and i > 0:
            print("saving " + save_file)
            trainable_gan.save()

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

if __name__ == "__main__":
    config = hg.configuration.Configuration.load(args.config+".json")

    if args.action == 'train':
        metrics = train(config, inputs, args)
        print("Resulting metrics:", metrics)
    elif args.action == 'sample':
        sample(config, inputs, args)
    else:
        print("Unknown action: "+args.action)
