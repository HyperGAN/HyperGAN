from .base_gan import BaseGAN
from hyperchamber import Config
from hypergan.discriminators import *
from hypergan.distributions import *
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.generators import *
from hypergan.inputs import *
from hypergan.samplers import *
from hypergan.trainers import *
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


class AlignedInterpolateGAN(BaseGAN):
    """ 
    """
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)

    def create(self):
        self.latent = self.create_component("latent")
        self.x = self.inputs.next()[0]
        if self.config.use_latent:
            self.generator = self.create_component("generator", input_shape=[self.latent.next().shape[1]])
        else:
            self.generator = self.create_component("generator", input=self.x)
        if self.config.shared_discriminator:
            self.discriminator = self.create_component("discriminator", context_shapes={"class": [1]})
        else:
            self.discriminator = self.create_component("discriminator")
            self.discriminator2 = self.create_component("discriminator")
        self.loss = self.create_component("loss")
        self.trainer = self.create_component("trainer")
        self.sigmoid = torch.nn.Sigmoid()
        self.gammas = [torch.Tensor([self.config.interpolate]).float()[0].cuda(), torch.Tensor([1.-self.config.interpolate]).float()[0].cuda()]

    def forward_discriminator(self, inputs):
        if self.config.shared_discriminator:
            d0_class = torch.zeros([inputs[0].shape[0], 1], device="cuda:0")
            d1_class = torch.ones([inputs[1].shape[0], 1], device="cuda:0")
            d0 = self.discriminator(inputs[0], context={"class": d0_class})
            d1 = self.discriminator(inputs[1], context={"class": d1_class})
            return d0 * self.gammas[0] * torch.sigmoid(d1) + d1 * self.gammas[1] * torch.sigmoid(d0)

        if self.config.union:
            d0 = self.discriminator(inputs[0])
            d1 = self.discriminator2(inputs[1])
            d3 = self.discriminator(inputs[1])
            d4 = self.discriminator2(inputs[0])
            return d0 * self.gammas[0] * torch.sigmoid(d1) + d1 * self.gammas[1] * torch.sigmoid(d0) + \
                    d3 * self.gammas[0] * torch.sigmoid(d4) + d4 * self.gammas[1] * torch.sigmoid(d3)

        else:
            d0 = self.discriminator(inputs[0])
            d1 = self.discriminator2(inputs[1])
            return self.sigmoid(d1)*d0*self.gammas[0]  + self.sigmoid(d0)*d1*self.gammas[1]

    def forward_pass(self):
        self.x = self.inputs.next()
        self.y = self.inputs.next(1)

        self.augmented_x = self.train_hooks.augment_x(self.x)
        self.augmented_y = self.train_hooks.augment_x(self.y)
        self.augmented_latent = self.train_hooks.augment_latent(self.latent.next())
        if self.config.use_latent:
            g = self.generator(self.augment_latent)
        else:
            g = self.generator(self.x)
        self.g = g
        self.augmented_g = self.train_hooks.augment_g(self.g)
        d_real = self.forward_discriminator([self.augmented_x, self.augmented_y])
        d_fake = self.forward_discriminator([self.augmented_g, self.augmented_g])
        self.d_fake = d_fake
        self.d_real = d_real
        return d_real, d_fake

    def discriminator_components(self):
        if self.config.shared_discriminator:
            return [self.discriminator]
        else:
            return [self.discriminator, self.discriminator2]

    def generator_components(self):
        return [self.generator]

    def discriminator_fake_inputs(self, discriminator_index=0):
        return [[self.augmented_g, self.augmented_g]]

    def discriminator_real_inputs(self, discriminator_index=0):
        if hasattr(self, 'y'):
            return [self.augmented_x, self.augmented_y]
        else:
            return [self.inputs.next(), self.inputs.next(1)]
