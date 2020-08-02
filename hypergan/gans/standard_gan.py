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

class StandardGAN(BaseGAN):
    """ 
    Standard GANs consist of:
    
    *required to sample*
    
    * latent
    * generator
    * sampler

    *required to train*

    * discriminator
    * loss
    * trainer
    """
    def __init__(self, *args, **kwargs):
        self.discriminator = None
        self.latent = None
        self.generator = None
        self.loss = None
        self.trainer = None
        self.features = []
        BaseGAN.__init__(self, *args, **kwargs)
        self.x = self.inputs.next()

    def build(self):
        torch.onnx.export(self.generator, torch.randn(*self.latent.z.shape, device='cuda'), "generator.onnx", verbose=True, input_names=["latent"], output_names=["generator"])

    def required(self):
        return "generator".split()

    def create(self):
        config = self.config

        self.latent = self.create_component("latent")
        self.generator = self.create_component("generator", input_shape=self.latent.shape)
        self.discriminator = self.create_component("discriminator")
        self.loss = self.create_component("loss")
        self.trainer = self.create_component("trainer")

    def forward_discriminator(self, inputs):
        return self.discriminator(inputs[0])

    def forward_pass(self):
        self.x = self.inputs.next()
        g = self.generator(self.latent.next())
        self.g = g
        d_real = self.forward_discriminator([self.x])
        d_fake = self.forward_discriminator([g])
        self.d_fake = d_fake
        self.d_real = d_real
        return d_real, d_fake

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
        return [self.generator]

    def discriminator_fake_inputs(self, discriminator_index=0):
        return [self.g]

    def discriminator_real_inputs(self, discriminator_index=0):
        if hasattr(self, 'x'):
            return [self.x]
        else:
            return [self.inputs.next()]

