from .base_gan import BaseGAN
from torchvision import transforms
from hyperchamber import Config
from hypergan.discriminators import *
from hypergan.distributions import *
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.generators import *
from hypergan.inputs import *
from hypergan.samplers import *
from hypergan.trainers import *
from torch.nn import functional as F
import copy
import hyperchamber as hc
import hypergan as hg
import importlib
import json
import numpy as np
import os
import sys
import time
import torch
import uuid
from hypergan.train_hooks.clip_prompt_train_hook import ClipPromptTrainHook, MakeCutouts
import open_clip

class ReptileGAN(BaseGAN):
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)
        self.x = self.inputs.next()
        self.step = 0

    def build(self):
        torch.onnx.export(self.generator, self.latent.next(), "generator.onnx", verbose=True, input_names=["latent"], output_names=["generator"], opset_version=11)

    def required(self):
        return "generator".split()

    def create(self):
        self.latent = self.create_component("latent")
        self.generator = self.create_component("generator", input=self.latent)
        #self.encoder = self.create_component("encoder")
        self.projector = self.create_component("projector")
        self.discriminator = self.create_component("discriminator")
        self.decoder = self.generator
        cut_size = 224
        self.make_cutouts = MakeCutouts(cut_size, self.config.cutn or 1, cut_pow=1)
        model, train_transform, eval_transform = open_clip.create_model_and_transforms(self.config.model or 'ViT-B-32', self.config.model_provider or 'openai')
        self.perceptor = model.eval().requires_grad_(False).to('cuda:0')

        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711])


    def forward_discriminator(self, *inputs):
        return self.discriminator(inputs[0])

    def next_inputs(self):
        self.x = self.inputs.next()
        self.augmented_latent = self.train_hooks.augment_latent(self.latent.next())
        self.augmented_x = self.train_hooks.augment_x(self.x)

    def forward_pass(self, task):
        x1 = self.augmented_x
        #e1 = self.encoder(x1)
        #e1 = torch.cat([e1, self.augmented_latent], dim=1)
        e1 = self.projector(self.augmented_latent)
        g = self.generator(e1)
        self.g = g
        self.augmented_g = self.train_hooks.augment_g(self.g)
        #ex = self.encode_image(self.augmented_x)
        #eg = self.encode_image(self.augmented_g)
        ex = self.augmented_x
        eg = self.augmented_g
        #if task == 0:
        #auto_x = -ex#self.partial_noise(ex)
        #auto_g = -eg#self.partial_noise(eg)
        auto_x = ex
        auto_g = eg
        #if task == 1:
        #    ex2 = self.encode_image(self.augmented_x)
        #    eg2 = self.encode_image(self.augmented_g)
        #auto_x = torch.cat([ex, ex2], dim=1)
        #auto_g = torch.cat([eg, eg2], dim=1)
        #auto_x = torch.cat([-ex, -ex])
        #auto_g = torch.cat([-eg, -eg])

        #if task == 0:
        #    auto_x = torch.cat([self.augmented_x, self.augmented_x])
        #    auto_g = torch.cat([self.augmented_g, self.augmented_g])
        #if task == 1:
        #    x1 = self.augmented_x
        #    self.next_inputs()
        #    auto_x = torch.cat([x1, self.augmented_x])
        #    auto_g = torch.cat([x1, self.augmented_g])


        #if task == 0:
        #    auto_x = torch.cat([self.augmented_x, self.augmented_x])
        #    auto_g = torch.cat([self.augmented_g, self.augmented_g])
        #if task == 1:
        #    auto_x = torch.cat([x1, self.augmented_x])
        #    auto_g = torch.cat([x1, self.augmented_g])
        d_fake = self.forward_discriminator(auto_g)
        d_real = self.forward_discriminator(auto_x)
        self.d_fake = d_fake
        self.d_real = d_real
        return d_real, d_fake

    def encode_image(self, x):
        return self.perceptor.encode_image(self.make_cutouts(self.normalize((x+1.0)/2.0)))

    def partial_noise(self, x):
        gamma = self.config.denoising_gamma or 0.2
        mask = (torch.rand_like(x) > gamma).float()
        return mask*x

    def forward_loss(self, task=0):
        d_real, d_fake = self.forward_pass(task)
        if task == 0:
            criterion = torch.nn.BCEWithLogitsLoss()
            cr = torch.mean(d_real,0)
            cf = torch.mean(d_fake,0)
            g_loss = criterion(d_fake-cr, torch.ones_like(d_fake))
            d_loss = criterion(d_real-cf, torch.ones_like(d_real)) + criterion(d_fake-cr, torch.zeros_like(d_fake))
        #reg = self.generator.context["reg"]
        #elif task == 1:
        #    g_loss = d_fake
        #    d_loss = d_real - d_fake
        if task == 1:
            self.forward_discriminator(self.x)#self.partial_noise(self.encode_image(self.x)))
            d = self.discriminator.context['z']
            regularizer = torch.mean(d, dim=1)**2
            regularizer = regularizer# + (torch.std(d, dim=1, unbiased=False)-torch.ones_like(regularizer))**2
            x_prime = self.generator(d)
            mse_loss = F.mse_loss(self.x, x_prime)#+ regularizer
            g_loss = mse_loss
            d_loss = mse_loss

        return [d_loss, g_loss]

    def input_nodes(self):
        "used in hypergan build"
        return [
        ]

    def output_nodes(self):
        "used in hypergan build"
        return [
        ]

    def discriminator_components(self):
        return [self.discriminator]

    def generator_components(self):
        return [self.generator, self.projector]#, self.encoder]

    def discriminator_fake_inputs(self):
        return [[self.augmented_g]]

    def discriminator_real_inputs(self):
        if hasattr(self, 'augmented_x'):
            return [self.augmented_x]
        else:
            return [self.inputs.next()]

