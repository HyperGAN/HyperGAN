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

    def build(self):
        torch.onnx.export(self.generator, self.latent.next(), "generator.onnx", verbose=True, input_names=["latent"], output_names=["generator"], opset_version=11)

    def required(self):
        return "generator".split()

    def create(self):
        self.latent = self.create_component("latent")
        words = LayerShape(32, 300)
        coord = LayerShape(1)
        self.x = self.inputs.next()[0]
        self.encoder = self.create_component("encoder", input=self.x)
        z_shape = self.encoder.layer_shape
        self.decoder = self.create_component("decoder", input=z_shape)
        self.generator = self.decoder
        self.discriminator = self.create_component("discriminator")
        if self.config.image_loss:
            self.discriminator2 = self.create_component("discriminator")

        def image_loss():
            d_fake2 = self.discriminator2(self.encoder(self.g))
            d_real2 = self.discriminator2(self.encoder(self.x))
            d_loss2, g_loss2 = self.loss.forward(d_real2, d_fake2)
            self.add_metric("d_loss2", d_loss2)
            self.add_metric("g_loss2", g_loss2)

            #fake_in = [self.x,self.g]
            #real_in = [self.x,self.x]
            #if self.target_x == None:
            #    self.target_g = [Parameter(g, requires_grad=True) for g in fake_in]
            #    self.target_x = [Parameter(x, requires_grad=True) for x in real_in]
            #for target, data in zip(self.target_x, real_in):
            #    target.data = data.clone()
            #for target, data in zip(self.target_g, fake_in):
            #    target.data = data.clone()
            #d_add, g_add = self.hooks[-2].do_hook(d_loss2, g_loss2, fake_in, real_in, self.forward_image_discriminator, d_fake2, d_real2, self.target_g, self.target_x)
            #lam = self.config.image_lambda or 1.0
            #d_loss2 += lam*d_add
            #g_loss2 += lam*g_add
            #self.add_metric("d_add", d_add)
            #self.add_metric("g_add", g_add)
            #for target, data in zip(self.target_x, real_in):
            #    target.data = data.clone()
            #for target, data in zip(self.target_g, fake_in):
            #    target.data = data.clone()

            #d_add, g_add = self.hooks[-1].do_hook(d_loss2, g_loss2, fake_in, real_in, self.forward_image_discriminator, d_fake2, d_real2, self.target_g, self.target_x)
            #d_loss2 += d_add
            #g_loss2 += g_add
            #self.add_metric("d_add2", d_add)
            #self.add_metric("g_add2", g_add)

            return [d_loss2, g_loss2]
        def mse_loss():
            lam = self.config.mse or 1.0
            loss = torch.nn.MSELoss()(self.g,self.x)
            self.add_metric("mse", loss * lam)
            return None, loss * lam

        if self.config.image_loss:
            self.add_loss(image_loss)
        else:
            self.add_loss(mse_loss)



    def forward_discriminator(self, inputs):
        return self.discriminator(torch.cat([inp.view(*self.encoding.shape) for inp in inputs], dim=1))
    def forward_image_discriminator(self, inputs):
        return self.image_discriminator(torch.cat([inputs[0], inputs[1]], dim=1), context={"words":inputs[2]})

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
        elif self.config.image_loss:
            return [self.discriminator, self.discriminator2]
        else:
            return [self.discriminator, self.discriminator2]

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

