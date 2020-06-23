# https://arxiv.org/pdf/1911.00888v1.pdf
from .base_gan import BaseGAN
import torch.nn as nn
from hyperchamber import Config
from hypergan.discriminators import *
from hypergan.distributions import *
from hypergan.distributions.uniform_distribution import UniformDistribution
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.generators import *
from hypergan.inputs import *
from hypergan.samplers import *
from hypergan.trainers import *
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
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

class MultiMarginalGAN(BaseGAN):
    """ 
    """
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)

    def required(self):
        """
        `input_encoder` is a discriminator.  It encodes X into Z
        `discriminator` is a standard discriminator.  It measures X, reconstruction of X, and G.
        `generator` produces two samples, input_encoder output and a known random distribution.
        """
        return "generator discriminator ".split()

    def create(self):
        config = self.config

        self.latent = self.create_component("latent")
        self.encoder = self.create_component("encoder")
        self.generators = [self.create_component("generator", input=self.encoder) for _d in self.inputs.datasets[1:]]
        self.generator = self.generators[0]
        self.discriminator = self.create_component("discriminator")
        self.loss = self.create_component("loss")
        self.trainer = self.create_component("trainer")

    def g_parameters(self):
        for gen in self.generators:
            for param in gen.parameters():
                yield param
        for param in self.encoder.parameters():
            yield param

    def d_parameters(self):
        return self.discriminator.parameters()

    def forward_discriminator(self, g, x_target):
        D = self.discriminator
        d_real = D(x_target)
        d_fake = D(g)
        return d_real, d_fake

    def forward_loss(self):
        x0 = self.inputs.next(0)
        x1 = self.inputs.next(1)

        g = self.generators[0](self.encoder(x0))
        d_real0, d_fake0 = self.forward_discriminator(g, x0)
        d_loss0, g_loss0 = self.loss.forward(d_real0, d_fake0)
        lambda0 = self.config.lambda0 or 1

        d_real1, d_fake1 = self.forward_discriminator(g, x1)
        d_loss1, g_loss1 = self.loss.forward(d_real1, d_fake1)
        lambdaN = self.config.lambdaN or 10

        d_loss = d_loss0 * lambda0 + d_loss1 * lambdaN
        g_loss = g_loss0 * lambda0 + g_loss1 * lambdaN

        self.d_fakes = [d_fake0, d_fake1]
        self.xs = [x0, x1]

        if self.config.vae:
            logvar = self.encoder.vae.sigma
            mu = self.encoder.vae.mu
            vae = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            #vae = (self.config.vae_lambda or 1) * vae
            self.add_metric('vae0', vae)


        if self.config.mse:
            E = self.encoder
            G = self.generators[0]
            inp = x1
            l1_loss = nn.MSELoss()(G(E(inp)),  inp)
            self.add_metric("l1", l1_loss)
            g_loss += l1_loss
        
        return d_loss, g_loss

