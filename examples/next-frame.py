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
        self.encoder = self.create_component("encoder")
        self.decoder = self.create_component("decoder")
        c_w = 8
        c_h = 8
        c_channels = 512
        c_shape = LayerShape(c_channels, c_h, c_w)
        self.state= self.create_component("state", context_shapes={"past": c_shape} )
        self.discriminator = self.create_component("discriminator")

    def forward_discriminator(self, inputs):
        return self.discriminator(inputs[0])

    def forward_pass(self, frames, xs, loss):
        d_fakes = []
        d_reals = []
        d_losses = []
        g_losses = []

        D = self.discriminator
        self.d_fake_inputs = []
        state = self.gen_state() 
        self.gs = frames
        for i in range(self.frames-self.per_sample_frames+1):
            rems = frames[i:i+self.per_sample_frames-1]
            enc = self.encoder(torch.cat(rems, dim=1))
            state = self.state(enc, context={"past": state})
            g = self.decoder(state)
            self.gs.append(g)
            rems += [g]
            xframes = torch.cat(frames[i:i+self.per_sample_frames], dim=1)
            d_real = D(xframes)
            self.d_real = d_real
            d_reals.append(d_real)
            d_fake_input = torch.cat(rems, dim=1)
            self.d_fake_inputs.append(d_fake_input.clone().detach())
            d_fake = D(d_fake_input)
            _d_loss, _g_loss = loss.forward(d_real, d_fake)
            d_losses.append(_d_loss)
            g_losses.append(_g_loss)
 
        d_loss = sum(d_losses)/len(d_losses)
        g_loss = sum(g_losses)/len(g_losses)
        self.d_fake = sum(d_fake)/len(d_fake)
        return d_loss, g_loss

    def forward_loss(self, loss=None):
        current_inputs = list(torch.chunk(self.inputs.next(), self.frames, dim=1))
        self.xs = current_inputs
        self.last_x = current_inputs[-1]

        if self.config.discriminator:
            d_loss, g_loss = self.forward_pass(current_inputs, self.xs, loss)

        return d_loss, g_loss

    def discriminator_components(self):
        components = [self.discriminator]
        return components

    def generator_components(self):
        components = [self.encoder, self.decoder, self.state]
        return components

    def channels(self):
        return super(NextFrameGAN, self).channels() * self.per_sample_frames

    def gen_state(self):
        shape = (self.batch_size(), self.encoder.current_size.channels, self.encoder.current_size.height, self.encoder.current_size.width)
        if self.config.statedist == "random":
            return self.random_z(self.latent.sample)
        if self.config.statedist == "uniform":
            return torch.rand(shape).cuda() * 2.0 - 1
        if self.config.statedist == "uniform_1_to_1":
            return torch.rand(shape).cuda() * 2.0 - 1
        if self.config.statedist == "zeros":
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
        self.refresh_input_cache()
        self.seed()
        self.i = 0

    def seed(self):
        #self.c = self.gan.gen_c()
        #self.z = self.gan.gen_z()
        #self.z = self.gan.random_z(self.gan.latent.sample)
        #self.c = self.gan.random_c(self.gan.latent.sample)
        g = self.input_cache[0]
        self.state = self.gan.gen_state()
        print("STATET1", self.state.shape)

        enc = self.gan.encoder(torch.cat(self.input_cache[:-1], dim=1))
        self.state = self.gan.state(enc, context={"past": self.state})
        print("STATET", self.state.shape)
        g = self.gan.decoder(self.state)
        self.current_frames = self.input_cache[:-1]+[g]
        self.g = g

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
        self.next_input()
        samples = []
        inframes = self.current_frames[-args.per_sample_frames+1:]
        enc = self.gan.encoder(torch.cat(inframes, dim=1))
        self.state = self.gan.state(enc, context={"past": self.state})
        g = self.gan.decoder(self.state)
        self.current_frames += [g]
        self.current_frames = self.current_frames[1:]
        self.g = g
        print(self.g.mean())
        if self.i % (8*24) == 0:
            print("RESET")
            self.seed()
        self.i += 1
        samples += [('generator', self.g)]
        return samples


class TrainingVideoFrameSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)

    def _sample(self):
        gan = self.gan
        return [('input', self.gan.last_x), ('gfirst', self.gan.gs[-2]), ('generator', self.gan.gs[-1])]



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
