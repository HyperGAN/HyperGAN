import importlib
import json
import numpy as np
import os
import sys
import time
import uuid
import copy

from hypergan.discriminators import *
from hypergan.distributions import *
from hypergan.generators import *
from hypergan.inputs import *
from hypergan.samplers import *
from hypergan.trainers import *

import hyperchamber as hc
import hypergan as hg

from hypergan.gan_component import ValidationException, GANComponent
from .base_gan import BaseGAN

from hypergan.distributions.uniform_distribution import UniformDistribution
from hypergan.trainers.experimental.consensus_trainer import ConsensusTrainer

class AliGAN(BaseGAN):
    """ 
    """
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)

    def create(self):
        self.classes_count = len(self.inputs.datasets)
        self.latent = self.create_component("latent")
        self.encoder = self.create_component("encoder")
        if self.classes_count == 2:
            self.encoder2 = self.create_component("encoder")
        self.generator = self.create_component("generator", input=self.encoder)
        self.discriminator = self.create_component("discriminator")
        self.loss = self.create_component("loss")
        self.trainer = self.create_component("trainer")

    def g_parameters(self):
        for param in self.generator.parameters():
            yield param
        for param in self.encoder.parameters():
            yield param
        if self.classes_count == 2:
            for param in self.encoder2.parameters():
                yield param

    def d_parameters(self):
        return self.discriminator.parameters()

    def forward_discriminator(self):
        E = self.encoder
        D = self.discriminator
        G = self.generator
        real = self.inputs.next()
        enc_real = E(real)
        G_sample = G(enc_real)

        if self.classes_count == 1:
            d_real = D(real, context={"z": enc_real})
            d_fake = D(G_sample, context={"z": enc_real})
            self.gp_context = {"z":enc_real}
            self.gp_context_x = {"z":enc_real}
            self.gp_context_g = {"z":enc_real}
        elif self.classes_count == 2:
            E2 = self.encoder2
            real1 = self.inputs.next(1)
            enc_real1 = E2(real1)
            d_real = D(real1, context={"z": enc_real1})
            d_fake = D(G_sample, context={"z": enc_real})
            self.gp_context_x = {"z":enc_real1}
            self.gp_context_g = {"z":enc_real}

        self.generator_sample = G_sample

        return d_real, d_fake

    def forward_loss(self):
        d_real, d_fake = self.forward_discriminator()
        d_loss, g_loss = self.loss.forward(d_real, d_fake)

        return d_loss, g_loss
