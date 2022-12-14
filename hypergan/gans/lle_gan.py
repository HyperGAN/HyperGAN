from .base_gan import BaseGAN
from hyperchamber import Config
from hypergan.discriminators import *
from hypergan.distributions import *
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.generators import *
from hypergan.inputs import *
from hypergan.samplers import *
from hypergan.trainers import *
from torch.nn import functional as F
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

class LLEGAN(BaseGAN):
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)
        self.x = self.inputs.next()

    def build(self):
        torch.onnx.export(self.generator, self.latent.next(), "generator.onnx", verbose=True, input_names=["latent"], output_names=["generator"], opset_version=11)

    def required(self):
        return "generator".split()

    def create(self):
        self.latent = self.create_component("latent")
        self.generator = self.create_component("generator", input=self.latent)
        self.discriminator = self.create_component("discriminator")
        self.softplus = torch.nn.Softplus(self.config.beta or 1, self.config.threshold or 20)

    def forward_discriminator(self, *inputs):
        return self.discriminator(inputs[0])

    def next_inputs(self):
        self.x = self.inputs.next()
        self.augmented_latent = self.train_hooks.augment_latent(self.latent.next())
        self.augmented_x = self.train_hooks.augment_x(self.x)

    def lle(self, label):
        outputs = [self.discriminator.context[x] for x in ["out0", "out1", "out2", "out3"]]
        outputs = torch.cat(outputs, dim=1)
        return outputs[range(outputs.shape[0]), label]

    def forward_pass(self):
        g = self.generator(self.augmented_latent)
        self.g = g
        self.augmented_g = self.train_hooks.augment_g(self.g)
        self.forward_discriminator(self.augmented_g)
        d_fake = self.discriminator.context["out0"]
        self.forward_discriminator(self.augmented_x)
        d_real = self.lle(self.inputs.data['target_class'])
        self.d_fake = d_fake
        self.d_real = d_real
        return d_real, d_fake

    def forward_loss(self):
        d_real, d_fake = self.forward_pass()
        dist_shift_label = self.inputs.data['labels'].to('cuda:0').float()
        class_mask = dist_shift_label[:, 0]
        dist_shift_output = self.discriminator.context["predictor"]

        d_loss = class_mask * self.softplus(-d_real) + class_mask * self.softplus(d_fake)
        g_loss = self.softplus(-d_fake)
        

        ce_loss = F.cross_entropy(dist_shift_output, dist_shift_label)

        self.add_metric('ce', ce_loss.sum())

        return [d_loss+ce_loss, g_loss]


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
            return [self.inputs.next()]

