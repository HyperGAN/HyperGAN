from .base_gan import BaseGAN
from hyperchamber import Config
from hypergan.train_hooks.clip_prompt_train_hook import ClipPromptTrainHook, MakeCutouts
import open_clip
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
        cut_size = 224
        self.make_cutouts = MakeCutouts(cut_size, self.config.cutn or 1, cut_pow=1)
        model, train_transform, eval_transform = open_clip.create_model_and_transforms(self.config.model or 'ViT-B-32', self.config.model_provider or 'openai')

        self.perceptor = model.eval().requires_grad_(False).to(self.device)
 
        def latent_ce_loss():
            target = self.actual_latent_ce
            prediction = self.predicted_latent
            loss = F.cross_entropy(prediction, target)# * 0.05
            self.add_metric('ce', loss)
            return loss, loss

        def kloss():
            prediction = self.predicted_x
            target = (self.encode_image(self.x) > 0.0 ).float()
            loss = F.cross_entropy(prediction, target)# * 0.05
            self.add_metric('cex', loss)
            self.add_metric('min', target.min().mean())
            self.add_metric('mean', target.mean())
            self.add_metric('max', target.max().mean())
            return loss, None
        def kloss2():

            prediction = self.predicted_text
            target = (self.perceptor.encode_text(open_clip.tokenize(self.text).to(self.device)).float() > 0.0).float()
            loss = F.cross_entropy(prediction, target)# * 0.05
            self.add_metric('cetxt', loss)
            self.add_metric('tmin', target.min().mean())
            self.add_metric('tmean', target.mean())
            self.add_metric('tmax', target.max().mean())
            return loss, None
 
        self.add_loss(latent_ce_loss)
        self.add_loss(kloss)
        self.add_loss(kloss2)
    def encode_image(self, img):
        return self.perceptor.encode_image(self.make_cutouts(img)).float()

    def forward_discriminator(self, *inputs):
        return self.discriminator(inputs[0], context={"text": inputs[1]})

    def next_inputs(self):
        inp = self.inputs.next()
        self.x = inp['img']
        self.text = inp['txt']
        self.augmented_latent = self.train_hooks.augment_latent(self.latent.next())
        self.augmented_x = self.train_hooks.augment_x(self.x)


    def forward_pass(self):
        b = self.x.shape[0]

        text_embeds = self.text_encoder.encode_text(self.text)
        self.text_embeds = text_embeds
        self.g = self.generator(self.augmented_latent, context={"text": text_embeds})
        self.augmented_g = self.train_hooks.augment_x(self.g)
        x_args = [self.augmented_x, text_embeds]
        g_args = [self.augmented_g, text_embeds]
        self.x_args = x_args
        self.g_args = g_args

        d_real = self.forward_discriminator(*x_args)
        self.predicted_x = self.discriminator.context["x_prediction"].clone()
        self.predicted_text = self.discriminator.context["text_prediction"].clone()
        d_fake = self.forward_discriminator(*g_args)
        self.predicted_latent = self.discriminator.context["z_prediction"]
        self.actual_latent_ce = (self.augmented_latent > 0).float()
        
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
