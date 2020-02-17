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
import torch.nn as nn
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

        if self.classes_count == 1:
            real = self.inputs.next()
            enc_real = E(real)
            g = G(enc_real)
            d_real = D(real, context={"z": enc_real})
            d_fake = D(g, context={"z": enc_real})
            self.g = g
            self.z = enc_real
            self.x = real
        elif self.classes_count == 2:
            E2 = self.encoder2
            real = self.inputs.next()
            enc_real = E2(real)
            g = G(enc_real)
            real1 = self.inputs.next(1)
            enc_real1 = E(real1)
            d_real = D(real1, context={"z": enc_real1})
            d_fake = D(g, context={"z": enc_real})
            self.z = enc_real1
            self.x = real1
            self.g = G(enc_real1)
        self.d_fake = d_fake

        return d_real, d_fake

    def forward_loss(self):
        """
            Runs a forward pass through the GAN and returns (d_loss, g_loss)
        """
        d_real, d_fake = self.forward_discriminator()
        d_loss, g_loss = self.loss.forward(d_real, d_fake)
        mse_criterion = nn.MSELoss()
        if self.config.vae:
            logvar = self.encoder.vae.sigma
            mu = self.encoder.vae.mu
            vae = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            #vae = (self.config.vae_lambda or 1) * vae
            self.add_metric('vae0', vae)

            if self.classes_count == 2:
                logvar = self.encoder2.vae.sigma
                mu = self.encoder2.vae.mu
                vae1 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                #vae1 = (self.config.vae_lambda or 1) * vae1
                vae += vae1
                self.add_metric('vae1', vae1)
 
            g_loss += vae
            if self.config.mse:
                mse = mse_criterion(self.g, self.x)
                self.add_metric('mse', mse)
                g_loss += mse

        return d_loss, g_loss


    def regularize_gradient_norm(self, calculate_loss):
        x = Variable(self.x, requires_grad=True).cuda()
        d1_logits = self.discriminator(x, context={"z":self.z})
        d2_logits = self.d_fake

        loss = calculate_loss(d1_logits, d2_logits)

        if loss == 0:
            return [None, None]

        d1_grads = torch_grad(outputs=loss, inputs=x, create_graph=True)
        d1_norm = [torch.norm(_d1_grads.reshape(-1).cuda(),p=2,dim=0) for _d1_grads in d1_grads]
        reg_d1 = [((_d1_norm**2).cuda()) for _d1_norm in d1_norm]
        reg_d1 = sum(reg_d1)

        return loss, reg_d1
