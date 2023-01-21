from .base_gan import BaseGAN
from hyperchamber import Config
from torch import nn
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
import random

class DdpgGAN(BaseGAN):
    """
    DDPG GANs consist of:

    * single input source
    * latent
    * generator
    * discriminator

    The generator creates a sample based on the latent.

    """
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)

    def build(self):
        torch.onnx.export(self.generator, self.latent.next(), "generator.onnx", verbose=True, input_names=["latent"], output_names=["generator"], opset_version=11)

    def required(self):
        return "generator".split()

    def create(self):
        self.softplus = torch.nn.Softplus(self.config.beta or 1, self.config.threshold or 20)
        self.latent = self.create_component("latent")
        self.generator_target = self.create_component("generator", input=self.latent)
        self.generator_local = self.create_component("generator", input=self.latent)
        self.generator = self.generator_local
        self.discriminator_target = self.create_component("discriminator")
        self.discriminator_local = self.create_component("discriminator")

        self.discriminator = self.discriminator_local
        self.memories = []
    def forward_discriminator(self, *inputs):
        return self.discriminator(inputs[0])

    def next_inputs(self):
        self.augmented_latent = self.train_hooks.augment_latent(self.latent.next())
        self.x = self.inputs.next()#gan=self)
        self.augmented_x = self.train_hooks.augment_x(self.x)

    def forward_pass(self):
        if self.memories == []:
            self.generator_target.load_state_dict(self.generator_local.state_dict())
            self.discriminator_target.load_state_dict(self.discriminator_local.state_dict())
        for x in self.x:
            self.memories.append(x.squeeze(0))
        length = self.x.size(0)*16
        if len(self.memories) > length:
            self.memories = self.memories[self.x.size(0):length]
        xs = random.sample(self.memories, self.x.size(0))
        self.x = torch.cat(xs, dim=0)
        g = self.generator(self.augmented_latent)
        self.g = g
        self.augmented_g = self.train_hooks.augment_g(self.g)
        d_fake = self.discriminator_local(self.augmented_g)
        d_real = self.discriminator_local(self.augmented_x)
        self.d_fake = d_fake
        self.d_real = d_real
        return d_real, d_fake

    def forward_loss(self, mode='gd'):
        if mode == 'g':
            d_real, d_fake = self.forward_pass()
            g_loss = self.softplus(-d_fake) + self.softplus(d_real)
            return None, g_loss
        d_real, d_fake = self.forward_pass()
        d_loss = self.softplus(-d_real) + self.softplus(d_fake)

        gamma = 0.1
        q_targets = d_loss + gamma * self.discriminator_target(self.generator_target(self.latent.next())).clone().detach()
        q_expected = d_real
        d_loss = torch.nn.functional.mse_loss(q_targets, q_expected)
        g_loss = None
        return [d_loss, g_loss]

    def post_step(self):
        self._tau = 0.001
        self.soft_update(self.discriminator_local, self.discriminator_target, self._tau)
        self.soft_update(self.generator_local, self.generator_target, self._tau)


    def soft_update(self, local_network: nn.Module, target_network: nn.Module, tau: float) -> None:
        for target_param, param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.mul_((1.0-tau))
            target_param.data.add_(tau* param.data)

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

    def discriminator_fake_inputs(self):
        return [[self.augmented_g]]

    def discriminator_real_inputs(self):
        if hasattr(self, 'augmented_x'):
            return [self.augmented_x]
        else:
            return [self.inputs.next(gan=self)]

