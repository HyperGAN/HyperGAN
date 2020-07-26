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
        self.generator = self.create_component("generator", input_size=[self.latent.next().shape[1]])
        self.discriminator = self.create_component("discriminator")
        self.discriminator2 = self.create_component("discriminator")
        self.loss = self.create_component("loss")
        self.trainer = self.create_component("trainer")
        self.sigmoid = torch.nn.Sigmoid()
        self.gammas = [torch.Tensor([self.config.interpolate]).float()[0].cuda(), torch.Tensor([1.-self.config.interpolate]).float()[0].cuda()]

    def forward_discriminator(self, inputs):
        d0 = self.discriminator(inputs[0])
        d1 = self.discriminator2(inputs[1])
        return self.sigmoid(d1)*d0*self.gammas[0]  + self.sigmoid(d0)*d1*self.gammas[1]

    def forward_pass(self):
        self.x = self.inputs.next()
        self.y = self.inputs.next(1)
        g = self.generator(self.latent.next())
        self.g = g
        d_real = self.forward_discriminator([self.x, self.y])
        d_fake = self.forward_discriminator([self.g, self.g])
        self.d_fake = d_fake
        self.d_real = d_real
        return d_real, d_fake

    def discriminator_components(self):
        return [self.discriminator, self.discriminator2]

    def generator_components(self):
        return [self.generator]

    def discriminator_fake_inputs(self, discriminator_index=0):
        return [self.g]

    def discriminator_real_inputs(self, discriminator_index=0):
        if hasattr(self, 'y'):
            return [self.x, self.y]
        else:
            return [self.inputs.next(), self.inputs.next(1)]
