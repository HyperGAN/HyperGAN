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
from hypergan.trainers.experimental.consensus_trainer import ConsensusTrainer
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

    def forward_discriminator(self, G, x):
        E = self.encoder
        D = self.discriminator
        d_real = D(x)
        d_fake = D(G(E(x)))
        return d_real, d_fake

    def forward_loss(self):
        x0 = self.inputs.next(0)
        x1 = self.inputs.next(1)

        d_real0, d_fake0 = self.forward_discriminator(self.generator, x0)
        d_loss0, g_loss0 = self.loss.forward(d_real0, d_fake0)
        lambda0 = self.config.lambda0 or 0.1

        d_real1, d_fake1 = self.forward_discriminator(self.generator, x1)
        d_loss1, g_loss1 = self.loss.forward(d_real1, d_fake1)
        lambdaN = 1.0 - lambda0

        d_loss = d_loss0 * lambda0 + d_loss1 * lambdaN
        g_loss = g_loss0 * lambda0 + g_loss1 * lambdaN

        self.d_fakes = [d_fake0, d_fake1]
        self.xs = [x0, x1]

        if self.config.l1_loss:
            E = self.encoder
            G = self.generators[0]
            inp = self.inputs.next(index = 1)
            l1_loss = nn.MSELoss()(G(E(inp)),  inp)
            self.add_metric("l1", l1_loss)
            g_loss += l1_loss

        return d_loss, g_loss

    def regularize_gradient_norm(self, calculate_loss):
        reg_d1 = []
        loss = 0.0
        for x_, d_fake in zip(self.xs[1:], self.d_fakes[1:]):
            x = Variable(x_, requires_grad=True).cuda()
            d1_logits = self.discriminator(x)
            d2_logits = d_fake

            loss += calculate_loss(d1_logits, d2_logits)

            d1_grads = torch_grad(outputs=loss, inputs=x, retain_graph=True, create_graph=True)
            d1_norm = [torch.norm(_d1_grads.view(-1).cuda(),p=2,dim=0) for _d1_grads in d1_grads]

            reg_d1 += [((_d1_norm**2).cuda()) for _d1_norm in d1_norm]
        reg_d1 = sum(reg_d1)

        return loss, reg_d1
