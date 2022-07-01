from .base_gan import BaseGAN
from hypergan.generators import legacy
from torchvision import transforms
from torch.nn import functional as F
from hypergan.losses.stable_gan_loss import StableGANLoss
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
from hypergan.layers.ntm import _split_cols
from hypergan.gan_component import GANComponent

class KnowledgeDistillationGAN(BaseGAN):
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
        self.discriminator = self.create_component("discriminator")
        with open(self.config.pkl, "rb") as f:
            G = legacy.load_network_pkl(f)['G_ema']
            G = G.eval().requires_grad_(False).to(self.device)
        self._teacher = G

    def forward_discriminator(self, *inputs):
        return self.discriminator(inputs[0])

    def next_inputs(self):
        pass


    def clip_encode(self, g):
        return PromptGAN.perceptor.encode_image(self.make_cutouts(self.normalize(g))).float()

    def teacher(self, z, class_idx=None, labels = None):
        G = self._teacher
        device = self.device
        batch_sz = self.batch_size()
        if G.c_dim != 0:
            # sample random labels if no class idx is given
            if class_idx is None:
                class_indices = np.random.randint(low=0, high=G.c_dim, size=(batch_sz))
                class_indices = torch.from_numpy(class_indices).to(device)
                w_avg = G.mapping.w_avg.index_select(0, class_indices)
            else:
                w_avg = G.mapping.w_avg[class_idx].unsqueeze(0).repeat(batch_sz, 1)
                class_indices = torch.full((batch_sz,), class_idx).to(device)

            if labels is None:
                labels = F.one_hot(class_indices, G.c_dim)

        else:
            w_avg = G.mapping.w_avg.unsqueeze(0)
            if labels is not None:
                labels = None
            if class_idx is not None:
                print('Warning: --class is ignored when running an unconditional network')

        w = self._teacher.mapping(z, labels)

        w_avg = w_avg.unsqueeze(1).repeat(1, G.mapping.num_ws, 1)
        truncation_psi = 1.0
        w = w_avg + (w - w_avg) * truncation_psi
        g = self._teacher.synthesis(w, noise_mode='const')
        return g, labels


    def forward_pass(self):
        prompt = self.config.prompt
        z = self.latent.next()
        x1, labels = self.teacher(z)
        self.labels = labels
        zlabels = torch.cat([z, labels], dim=1)
        g1 = self.generator(zlabels)
        self.x = x1
        xarg = torch.cat([x1,x1],dim=1)
        garg = torch.cat([x1,g1],dim=1)
        self.x_args = [xarg]
        self.g_args = [garg]
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
        return [self.discriminator]#, self.words_discriminator]

    def generator_components(self):
        return [self.generator]

    def discriminator_fake_inputs(self):
        return [self.g_args]

    def discriminator_real_inputs(self):
        return self.x_args
