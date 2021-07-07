from hyperchamber import Config
from hypergan.discriminators import *
from hypergan.distributions import *
from hypergan.gan_component import ValidationException, GANComponent
from .base_gan import BaseGAN
from hypergan.generators import *
from hypergan.inputs import *
from hypergan.layer_shape import LayerShape
from hypergan.samplers import *
from hypergan.trainers import *
from hypergan.losses.stable_gan_loss import StableGANLoss
from torch.nn.parameter import Parameter
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

class AAEGAN(BaseGAN):
    """
        https://arxiv.org/abs/1511.05644
    """
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)
        self.target_x = None
        self.stable_gan_loss = None

    def build(self):
        torch.onnx.export(self.generator, self.latent.next(), "generator.onnx", verbose=True, input_names=["latent"], output_names=["generator"], opset_version=11)

    def required(self):
        return "generator".split()

    def create(self):
        self.latent = self.create_component("latent")
        coord = LayerShape(1)
        self.x = self.inputs.next()[0]
        self.encoder = self.create_component("encoder", input=self.x)
        z_shape = self.encoder.layer_shape
        self.decoder = self.create_component("decoder", input=z_shape)
        self.generator = self.decoder
        self.discriminator = self.create_component("discriminator")
        if self.config.image_discriminator:
            self.image_discriminator = self.create_component("image_discriminator")

        def image_loss():
            ing = torch.cat([self.g,self.x], axis=1)
            inx = torch.cat([self.x,self.x], axis=1)
            if self.stable_gan_loss is None:
                self.stable_gan_loss = StableGANLoss(self.image_discriminator)
            d_loss2, g_loss2 = self.stable_gan_loss.stable_loss(inx, ing)
            self.add_metric("d_loss2", d_loss2)
            self.add_metric("g_loss2", g_loss2)

            return [d_loss2, g_loss2]
        def mse_loss():
            lam = self.config.mse or 1.0
            loss = torch.nn.MSELoss()(self.g,self.x)
            self.add_metric("mse", loss * lam)
            return None, loss * lam

        if self.config.image_discriminator:
            self.add_loss(image_loss)
        else:
            self.add_loss(mse_loss)



    def forward_discriminator(self, inputs):
        return self.discriminator(torch.cat([inp.view(*self.encoding.shape) for inp in inputs], dim=1))
    def forward_image_discriminator(self, inputs):
        return self.image_discriminator(torch.cat([inputs[0], inputs[1]], dim=1), context={})

    def next_inputs(self):
        self.x = self.inputs.next()
        self.x = self.x.to('cuda:0')
        self.augmented_latent = self.train_hooks.augment_latent(self.latent.next())
        self.augmented_x = self.train_hooks.augment_x(self.x)

    def forward_pass(self):
        self.encoding = self.encoder(self.x)
        self.g = self.decoder(self.encoding)
        self.augmented_g = self.train_hooks.augment_g(self.g)
        aug1 = self.train_hooks.augment_x(self.x)
        aug2 = self.train_hooks.augment_x(self.x)
        if self.config.image_discriminator:
            d_fake = self.forward_discriminator([self.encoding])
            d_real = self.forward_discriminator([self.augmented_latent])
        elif self.config.discriminator_exclude_encoding:
            d_fake = self.forward_discriminator([self.encoding])
            d_real = self.forward_discriminator([self.augmented_latent])
        else:
            d_fake = self.forward_discriminator([self.encoding])
            d_real = self.forward_discriminator([self.augmented_latent])
        self.d_fake = d_fake
        self.d_real = d_real
        return d_real, d_fake

    def forward_loss(self, loss):
        self.loss = loss
        d_real, d_fake = self.forward_pass()
        d_loss, g_loss = loss.forward(d_real, d_fake)

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
        if self.config.image_discriminator:
            return [self.discriminator, self.image_discriminator]
        else:
            return [self.discriminator]

    def generator_components(self):
        return [self.encoder, self.decoder]

    def discriminator_fake_inputs(self):
        if self.config.image_discriminator:
            return [[self.encoding]]
        elif self.config.discriminator_exclude_encoding:
            return [[self.encoding]]
        else:
            return [[self.encoding]]

    def discriminator_real_inputs(self):
        if self.config.image_discriminator:
            return [self.augmented_latent]
        elif self.config.discriminator_exclude_encoding:
            return [self.augmented_latent]
        else:
            return [self.augmented_latent]

