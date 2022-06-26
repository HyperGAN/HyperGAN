from .base_gan import BaseGAN
from torch.nn import functional as F
import open_clip
from hyperchamber import Config
from hypergan.discriminators import *
from hypergan.distributions import *
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.generators import *
from hypergan.inputs import *
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
from hypergan.train_hooks.clip_prompt_train_hook import ClipPromptTrainHook, MakeCutouts

class PretrainedKnowledgeGAN(BaseGAN):
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
        cut_size = 224
        self.make_cutouts = MakeCutouts(cut_size, self.config.cutn or 1, cut_pow=1)
        model, train_transform, eval_transform = open_clip.create_model_and_transforms(self.config.model or 'ViT-B-32', self.config.model_provider or 'openai')

        self.perceptor = model.eval().requires_grad_(False).to(self.device)
 

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
    def forward_discriminator(self, *inputs):
        return self.discriminator(inputs[0])

    def encode_image(self, img):
        return self.perceptor.encode_image(self.make_cutouts(img)).float()
    def next_inputs(self):
        self.x = self.inputs.next()
        self.augmented_latent = self.train_hooks.augment_latent(self.latent.next())
        self.augmented_x = self.train_hooks.augment_x(self.x)

    def forward_pass(self):
        g = self.generator(self.augmented_latent)
        self.g = g
        self.augmented_g = self.train_hooks.augment_g(self.g)
        d_real = self.forward_discriminator(self.augmented_x)
        self.predicted_x = self.discriminator.context["x_prediction"].clone()
        d_fake = self.forward_discriminator(self.augmented_g)
        self.d_fake = d_fake
        self.d_real = d_real
        return d_real, d_fake

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

