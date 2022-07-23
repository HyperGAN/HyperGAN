from .base_gan import BaseGAN
from torch.nn import functional as F
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
import open_clip
from hypergan.train_hooks.clip_prompt_train_hook import ClipPromptTrainHook, MakeCutouts
from hyperchamber import Config
from hypergan.discriminators import *
from hypergan.distributions import *
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.generators import *
from hypergan.inputs import *
from hypergan.layer_shape import LayerShape
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


class AutoencoderGAN(BaseGAN):
    """ 
    """
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)

    def create(self):
        self.latent = self.create_component("latent")
        self.x = self.inputs.next()
        self.encoder = self.create_component("encoder", input=self.x)
        self.generator = self.create_component("generator", input=self.encoder)
        e_shape = list(self.encoder.layer_shape().dims)
        e_shape[0] *= 2
        self.discriminator = self.create_component("discriminator", context_shapes={"z": LayerShape(*e_shape)})
        self.decoder = self.generator
        model, train_transform, eval_transform = open_clip.create_model_and_transforms(self.config.model or 'ViT-B-32', self.config.model_provider or 'openai')
        cut_size = 224
        self.make_cutouts = MakeCutouts(cut_size, self.config.cutn or 1, cut_pow=1)
        self.perceptor = model.eval().requires_grad_(False).to(self.device)
        self.quantizer = VectorQuantizer(self.config.n_embed or 256, 8*8, beta=0.25, remap=None, sane_index_shape=False).to(self.device)
        self.add_component("quantizer", self.quantizer)
        def add_emb_loss():
            return self.emb_loss, self.emb_loss

        def kloss():
            prediction = self.predicted_x
            target = (self.encode_image(self.x) > 0.0 ).float()
            loss = F.cross_entropy(prediction, target)# * 0.05
            self.add_metric('cex', loss)
            self.add_metric('min', target.min().mean())
            self.add_metric('mean', target.mean())
            self.add_metric('max', target.max().mean())
            return loss, None
        self.add_loss(kloss)
        self.add_loss(add_emb_loss)


    def encode(self, x):
        z, *_ = self.quantizer(self.encoder(x))
        return z


    def encode_image(self, img):
        return self.perceptor.encode_image(self.make_cutouts(img)).float()
    def forward_discriminator(self, *inputs):
        if self.config.z_ae:
            return self.discriminator(inputs[0], context={'z': inputs[1]})
        else:
            return self.discriminator(inputs[0])

    def next_inputs(self):
        self.x = self.inputs.next()

    def forward_pass(self):
        self.augmented_latent = self.train_hooks.augment_latent(self.latent.next())
        b = self.x.shape[0]
        aug1 = self.train_hooks.augment_x(self.x)
        aug2 = self.train_hooks.augment_x(self.x)
        self.augmented_x = [aug1, aug2]

        self.e, self.emb_loss, info = self.quantizer(self.encoder(self.x))
        self.add_metric('emb', self.emb_loss.mean())
        self.g = self.train_hooks.augment_g(self.generator(self.e))
        self.add_metric("l2", ((self.x - self.g)**2).mean())
        self.augmented_g = [aug1, self.g]
        x_args = self.augmented_x
        g_args = self.augmented_g
        self.x_args = x_args
        self.g_args = g_args

        d_fake = self.forward_discriminator(*g_args)
        d_real = self.forward_discriminator(*x_args)
        self.predicted_x = self.discriminator.context["x_prediction"]
        self.d_fake = d_fake
        self.d_real = d_real

        return d_real, d_fake

    def discriminator_components(self):
        return [self.discriminator]

    def generator_components(self):
        return [self.generator, self.encoder, self.quantizer]

    def discriminator_fake_inputs(self):
        return [self.g_args]

    def discriminator_real_inputs(self):
        return self.x_args
