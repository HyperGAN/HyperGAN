from .base_gan import BaseGAN
from hyperchamber import Config
from hypergan.discriminators import *
from hypergan.distributions import *
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.generators import *
from hypergan.inputs import *
from hypergan.layer_shape import LayerShape
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


class AutoencoderGAN(BaseGAN):
    """ 
    """
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)

    def create(self):
        self.latent = self.create_component("latent")
        self.x = self.inputs.next()
        self.encoder = self.create_component("encoder", input=self.x)
        self.generator = self.create_component("generator", input=self.encoder)
        e_shape = list(self.encoder.layer_shape().dims)
        e_shape[0] *= 2
        self.discriminator = self.create_component("discriminator", context_shapes={"z": LayerShape(*e_shape)})
        self.decoder = self.generator

    def forward_discriminator(self, *inputs):
        if self.config.z_ae:
            return self.discriminator(inputs[0], context={'z': inputs[1]})
        else:
            return self.discriminator(inputs[0])

    def next_inputs(self):
        self.x = self.inputs.next()

    def forward_pass(self):
        self.augmented_latent = self.train_hooks.augment_latent(self.latent.next())
        b = self.x.shape[0]
        aug1 = self.train_hooks.augment_x(self.x)
        aug2 = self.train_hooks.augment_x(self.x)
        self.augmented_x = torch.cat([aug1, aug2], dim=1)

        self.e = self.encoder(self.x)
        self.g = self.generator(self.e)
        g_aug = self.generator(self.augmented_latent)
        self.augmented_g = torch.cat([aug1, self.train_hooks.augment_g(self.g)], dim=1)
        e_real = self.augmented_latent.view(self.e.shape)
        if self.config.z_ae:
            z = torch.cat([e_real, self.e], dim=1)
            zprime = torch.cat([e_real, e_real], dim=1)
            x_args = [self.augmented_x, zprime]
            g_args = [self.augmented_g, z]
        else:
            x_args = [self.augmented_x]
            g_args = [self.augmented_g]
        self.x_args = x_args
        self.g_args = g_args

        d_fake = self.forward_discriminator(*g_args)
        d_real = self.forward_discriminator(*x_args)
        self.d_fake = d_fake
        self.d_real = d_real

        return d_real, d_fake

    def discriminator_components(self):
        return [self.discriminator]

    def generator_components(self):
        return [self.generator, self.encoder]

    def discriminator_fake_inputs(self):
        return [self.g_args]

    def discriminator_real_inputs(self):
        return self.x_args
