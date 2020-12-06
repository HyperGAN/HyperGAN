from .base_gan import BaseGAN
from hyperchamber import Config
from hypergan.discriminators import *
from hypergan.distributions import *
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.generators import *
from hypergan.inputs import *
from hypergan.samplers import *
from hypergan.layer_shape import LayerShape
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

    * single input source
    * latent
    * generator
    * discriminator

    The generator creates a sample based on the latent.

    """
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)
        self.x = self.inputs.next()
        self.classification = 0

    def build(self):
        torch.onnx.export(self.generator, torch.randn(*self.latent.z.shape, device='cuda'), "generator.onnx", verbose=True, input_names=["latent"], output_names=["generator"])

    def required(self):
        return "generator".split()

    def create(self):
        self.latent = self.create_component("latent")
        self.generator = self.create_component("generator", input=self.latent, context_shapes={"y": LayerShape(1)})
        self.discriminator = self.create_component("discriminator")

    def forward_discriminator(self, inputs):
        return self.discriminator(inputs[0])

    def forward_pass(self):
        if len(self.inputs.datasets) > 1:
            self.classification = self.classification % len(self.inputs.datasets)
        self.x = self.inputs.next(self.classification)
        self.augmented_latent = self.train_hooks.augment_latent(self.latent.next())
        b = self.x.shape[0]
        y_ = torch.randint(0, len(self.inputs.datasets), (b, )).to(self.x.device)
        g = self.generator(self.augmented_latent, context={"y": y_.float().view(b, 1)})
        self.latent_y = y_
        self.g = g
        self.augmented_x = self.train_hooks.augment_x(self.x)
        self.augmented_g = self.train_hooks.augment_g(self.g)
        d_real = self.forward_discriminator([self.augmented_x])
        d_fake = self.forward_discriminator([self.augmented_g])
        self.d_fake = d_fake
        self.d_real = d_real
        self.classification += 1
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
        return [self.augmented_g]

    def discriminator_real_inputs(self, discriminator_index=0):
        if hasattr(self, 'augmented_x'):
            return [self.augmented_x]
        else:
            return [self.inputs.next()]

