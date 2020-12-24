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


class AlignedGAN(BaseGAN):
    """ 
    """
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)

    def create(self):
        self.latent = self.create_component("latent")
        self.x = self.inputs.next()[0]
        if self.config.ali:
            self.encoder = self.create_component("encoder", input=self.x, context_shapes={"y": LayerShape(1)})
            self.generator = self.create_component("generator", input=self.encoder, context_shapes={"y": LayerShape(1)})
            self.discriminator = self.create_component("discriminator", context_shapes={"z": self.encoder.layer_shape()})
        else:
            self.generator = self.create_component("generator", input=self.x, context_shapes={"y": LayerShape(1)})
            self.discriminator = self.create_component("discriminator")

        self.classification = 0

    def forward_discriminator(self, inputs):
        if self.config.ali:
            return self.discriminator(inputs[0], context={"z":inputs[1]})
        else:
            return self.discriminator(inputs[0])

    def forward_pass(self):
        self.x = self.inputs.next()
        self.y = self.inputs.next(1)
        self.augmented_latent = self.train_hooks.augment_latent(self.latent.next())
        b = self.x.shape[0]
        self.augmented_x = self.train_hooks.augment_x(self.x)
        self.augmented_y = self.train_hooks.augment_x(self.y)

        if self.config.ali:
            encoding_x = self.encoder(self.x)
            encoding_y = self.encoder(self.y)
            self.encoding_y = encoding_y
            self.encoding_x = encoding_x
            self.g = self.generator(encoding_x)
            self.augmented_g = self.train_hooks.augment_g(self.g)
            x_args = [self.augmented_x, encoding_y]
            g_args = [self.augmented_g, encoding_x]
            y_args = [self.augmented_y, encoding_y]

        else:
            self.g = self.generator(self.x, context={"y": y_.float().view(b, 1)})
            self.augmented_g = self.train_hooks.augment_g(self.g)
            x_args = [self.augmented_x]
            g_args = [self.augmented_g]
            y_args = [self.augmented_y]

        if self.config.interpolate:
            d_real = self.forward_discriminator(x_args)
            d_real *= (1 - self.config.interpolate)
            d_real += self.forward_discriminator(y_args) * (self.config.interpolate)
        else:
            d_real = self.forward_discriminator(y_args)
        d_fake = self.forward_discriminator(g_args)
        self.d_fake = d_fake
        self.d_real = d_real
        self.classification += 1
        return d_real, d_fake

    def forward_loss(self, loss):
        d_real, d_fake = self.forward_pass()
        d_loss, g_loss = loss.forward(d_real, d_fake)
        if self.config.mse:
            lam = self.config.mse
            loss = torch.nn.MSELoss()(self.g,self.x.to(self.g.device))
            d_loss += loss * lam
            g_loss += loss * lam
            self.add_metric("mse", loss * lam)

        return [d_loss, g_loss]

    def discriminator_components(self):
        return [self.discriminator]

    def generator_components(self):
        return [self.generator]

    def discriminator_fake_inputs(self):
        if self.config.ali:
            return [[self.augmented_g, self.encoding_x]]
        else:
            return [[self.augmented_g]]

    def discriminator_real_inputs(self):
        if hasattr(self, 'y'):
            return [self.y, self.encoding_y]
        elif hasattr(self, 'encoder'):
            y = self.inputs.next(1)
            return [y, self.encoder(y)]
        else:
            y = self.inputs.next(1)
            return [y]
