from .base_gan import BaseGAN
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
        self.x = self.inputs.next()
        enc_real = E(self.x)
        G_sample = G(enc_real)

        if self.classes_count == 1:
            d_real = D(self.x, context={"z": enc_real})
            d_fake = D(G_sample, context={"z": enc_real})
            self.z = enc_real
        elif self.classes_count == 2:
            E2 = self.encoder2
            real1 = self.inputs.next(1)
            enc_real1 = E2(real1)
            d_real = D(real1, context={"z": enc_real1})
            d_fake = D(G_sample, context={"z": enc_real})
            self.z = enc_real1
        self.d_fake = d_fake

        return d_real, d_fake

    def regularize_gradient_norm(self, calculate_loss):
        x = Variable(self.x, requires_grad=True).cuda()
        z = Variable(self.z, requires_grad=True).cuda()
        d1_logits = self.discriminator(x, context={"z":z})
        d2_logits = self.d_fake

        loss = calculate_loss(d1_logits, d2_logits)

        if loss == 0:
            return [None, None]

        d1_grads = torch_grad(outputs=loss, inputs=x, retain_graph=True, create_graph=True)
        d1_z_grads = torch_grad(outputs=loss, inputs=z, retain_graph=True)

        d1_norm = [torch.norm(_d1_grads.view(-1).cuda(),p=2,dim=0) for _d1_grads in d1_grads]
        d1_z_norm = [torch.norm(_d1_grads.reshape(-1).cuda(),p=2,dim=0) for _d1_grads in d1_z_grads]

        reg_d1 = [((_d1_norm**2).cuda()) for _d1_norm in d1_norm]
        reg_d1 += [((_d1_norm**2).cuda()) for _d1_norm in d1_z_norm]
        reg_d1 = sum(reg_d1)

        return loss, reg_d1
