from .base_gan import BaseGAN
from hyperchamber import Config
from hypergan.discriminators import *
from hypergan.distributions import *
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

    def required(self):
        return "generator".split()

    def create(self):
        config = self.config

        self.latent = self.create_component("latent")
        self.generator = self.create_component("generator", input=self.latent)
        self.discriminator = self.create_component("discriminator")
        self.loss = self.create_component("loss")
        self.trainer = self.create_component("trainer")

    def forward_discriminator(self):
        self.x = self.inputs.next()
        g = self.generator(self.latent.sample())
        D = self.discriminator
        d_real = D(self.x)
        d_fake = D(g)
        self.d_fake = d_fake
        return d_real, d_fake

    def input_nodes(self):
        "used in hypergan build"
        return [
        ]

    def output_nodes(self):
        "used in hypergan build"
        return [
        ]

    def g_parameters(self):
        return self.generator.parameters()

    def d_parameters(self):
        return self.discriminator.parameters()

    def regularize_gradient_norm(self, calculate_loss):
        x = Variable(self.x, requires_grad=True).cuda()
        d1_logits = self.discriminator(x)
        d2_logits = self.d_fake

        loss = calculate_loss(d1_logits, d2_logits)

        if loss == 0:
            return [None, None]

        d1_grads = torch_grad(outputs=loss, inputs=x, retain_graph=True, create_graph=True)
        d1_norm = [torch.norm(_d1_grads.view(-1).cuda(),p=2,dim=0) for _d1_grads in d1_grads]

        reg_d1 = [((_d1_norm**2).cuda()) for _d1_norm in d1_norm]
        reg_d1 = sum(reg_d1)

        return loss, reg_d1
