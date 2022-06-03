from .base_gan import BaseGAN
from hyperchamber import Config
from hypergan.discriminators import *
from hypergan.distributions import *
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.generators import *
from hypergan.inputs import *
from hypergan.layer_shape import LayerShape
from hypergan.samplers import *
from hypergan.trainers import *
from hypergan.encoders.t5_text_encoder import T5TextEncoder
from torch import nn
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


class TextToImageGAN(BaseGAN):
    """ 
    """
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)

    def create(self):
        self.t5_dim = 64
        self.latent = self.create_component("latent")
        self.x = self.inputs.next()
        self.generator = self.create_component("generator", context_shapes={"text": LayerShape([self.t5_dim])})
        self.discriminator = self.create_component("discriminator", context_shapes={"text": LayerShape([self.t5_dim])})
        self.text_encoder = T5TextEncoder()
        self.decoder = self.generator

    def forward_discriminator(self, *inputs):
        return self.discriminator(inputs[0], context={"text": inputs[1]})

    def next_inputs(self):
        inp = self.inputs.next()
        self.x = inp['img']
        self.text = inp['txt']


    def forward_pass(self):
        b = self.x.shape[0]
        #lowres_x = self.upsample(self.x)
        #print("--", self.text)

        text_embeds = self.text_encoder.encode_text(self.text)
        self.g = self.generator(self.latent.next(), context={"text": text_embeds})
        x_args = [self.x, text_embeds]
        g_args = [self.g, text_embeds]
        self.x_args = x_args
        self.g_args = g_args

        d_fake = self.forward_discriminator(*g_args)
        d_real = self.forward_discriminator(*x_args)
        self.d_fake = d_fake
        self.d_real = d_real

        return d_real, d_fake

    def discriminator_components(self):
        return [self.discriminator]

    def generator_components(self):
        return [self.generator, self.text_encoder]

    def discriminator_fake_inputs(self):
        return [self.g_args]

    def discriminator_real_inputs(self):
        return self.x_args
