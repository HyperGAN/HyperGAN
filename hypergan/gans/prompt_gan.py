from .base_gan import BaseGAN
from torchvision import transforms
from torch.nn import functional as F
from hyperchamber import Config
from hypergan.discriminators import *
from hypergan.distributions import *
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.generators import *
from hypergan.inputs import *
from hypergan.samplers import *
from CLIP import clip
from torch import nn
from hypergan.trainers import *
from hypergan.layer_shape import LayerShape
from hypergan.train_hooks.clip_prompt_train_hook import ClipPromptTrainHook, MakeCutouts
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

class PromptGAN(BaseGAN):
    perceptor = None
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)
        self.x = self.inputs.next()
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711])

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    def build(self):
        torch.onnx.export(self.generator, self.latent.next(), "generator.onnx", verbose=True, input_names=["latent"], output_names=["generator"], opset_version=11)

    def required(self):
        return "generator".split()

    def create(self):
        self.latent = self.create_component("latent")
        self.generator = self.create_component("generator", input=self.latent)
        encoded_shape = LayerShape(1024)
        self.discriminator = self.create_component("discriminator", context_shapes={"encoded": encoded_shape})


    def forward_discriminator(self, *inputs):
        if len(inputs)>1:
            return self.discriminator(inputs[0], context= {"encoded":inputs[1]})
        else:
            return self.discriminator(inputs[0])

    def next_inputs(self):
        self.x = self.inputs.next()
        self.augmented_latent = self.train_hooks.augment_latent(self.latent.next())

    def forward_pass(self):
        prompt = self.config.prompt or ""
        if PromptGAN.perceptor is None:
            clip_model = (self.config.clip_model or "ViT-B/32")
            jit = True if float(torch.__version__[:3]) < 1.8 else False
            PromptGAN.perceptor = clip.load(clip_model, jit=jit)[0].eval().requires_grad_(False).to(self.device)
            self.encoded_text = PromptGAN.perceptor.encode_text(clip.tokenize(prompt).to(self.device)).float()
            self.encoded_text = self.encoded_text.expand([self.x.shape[0]]+list(self.encoded_text.shape[1:]))

            cut_size = PromptGAN.perceptor.visual.input_resolution
            self.make_cutouts = MakeCutouts(cut_size, self.config.cutn or 1, cut_pow=1)
        g = self.generator(self.augmented_latent)
        self.g = g
        self.augmented_g = self.augment_g(self.g)
        self.augmented_x = self.augment_x(self.x)
        enc_x = PromptGAN.perceptor.encode_image(self.normalize(self.make_cutouts(self.x))).float()
        enc_g = PromptGAN.perceptor.encode_image(self.normalize(self.make_cutouts(self.g))).float()
        self.x_args = [enc_x]
        self.g_args = [enc_g]
        d_real = self.forward_discriminator(*self.x_args)
        d_fake = self.forward_discriminator(*self.g_args)

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
        return [self.g_args]

    def discriminator_real_inputs(self):
        return self.x_args

