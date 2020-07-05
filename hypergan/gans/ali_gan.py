from .base_gan import BaseGAN
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
        if self.config.uz:
            self.uz = self.create_component("uz")
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
        if self.config.uz:
            for param in self.uz.parameters():
                yield param
        if self.classes_count == 2:
            for param in self.encoder2.parameters():
                yield param

    def d_parameters(self):
        return self.discriminator.parameters()

    def forward_pass(self):
        E = self.encoder
        G = self.generator
        D = self.discriminator

        if self.classes_count == 1:
            x0 = self.inputs.next()
            ex0 = E(x0)
            g = G(ex0)
            d_real = D(x0, context={"z": ex0})
            d_fake = D(x0, context={"z": ex0})
            self.g = g
            self.z = ex0
            self.x = x0
        elif self.classes_count == 2:
            E2 = self.encoder2
            x0 = self.inputs.next()
            x1 = self.inputs.next(1)
            if self.config.uz:
                UZ = self.uz
                euz = UZ(self.latent.sample())
                guz = G(euz)
            else:
                ex0 = E2(x0)
                gx1 = G(ex0, context=E2.context)

            ex1 = E(x1)
            if self.config.form == 1:
                d_real = D(x1, context={"z": ex1})
                d_fake = D(gx1, context={"z": ex0})
                self.z = ex1
                self.x = x1
            elif self.config.form == 3:
                d_real = D(x1, context={"z": ex1})
                d_fake = D(gx1, context={"z": E(gx1)})
                self.z = ex1
                self.x = x1
            elif self.config.form == 2:
                d_real = D(x1, context={"z": torch.abs(torch.randn_like(ex1))})
                d_fake = D(gx1, context={"z": ex0})
            else:
                d_real = D(x1, context={"z": ex0})
                d_fake = D(gx1, context={"z": ex1})
                self.z = ex0
                self.x = x1

            #self.g = G(enc_real1)
        self.d_fake = d_fake

        return d_real, d_fake

    def forward_loss(self):
        """
            Runs a forward pass through the GAN and returns (d_loss, g_loss)
        """
        d_real, d_fake = self.forward_pass()
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
            E = self.encoder
            G = self.generator
            mse = mse_criterion(G(E(self.x)), self.x)
            self.add_metric('mse', mse)
            g_loss += mse

        return d_loss, g_loss

