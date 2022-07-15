from .base_gan import BaseGAN
from hyperchamber import Config
from hypergan.discriminators import *
from hypergan.distributions import *
from hypergan.encoders.t5_text_encoder import T5TextEncoder
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.generators import *
from hypergan.inputs import *
from hypergan.layer_shape import LayerShape
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import copy
import hyperchamber as hc
import hypergan as hg
import importlib
import json
import kornia.augmentation as K
import numpy as np
import os
import random
import sys
import time
import torch
import uuid

class TextJepaGAN(BaseGAN):
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)
        self.x = self.inputs.next()

    def required(self):
        return "generator pred custom_encoder custom_text_encoder project_image project_text discriminator".split()

    def create(self):
        self.latent = self.create_component("latent")
        self.generator = self.create_component("generator", input=self.latent)
        self.custom_encoder = self.create_component("custom_encoder")
        self.project_image = self.create_component("project_image")
        self.project_text = self.create_component("project_text")
        self.custom_text_encoder = self.create_component("custom_text_encoder")
        self.pred = self.create_component("pred")
        encoded_shape = LayerShape(1024)
        self.discriminator = self.create_component("discriminator", context_shapes={"encoded": encoded_shape})

        def l2_loss():
            target = self.x
            prediction = self.autoencoding
            loss = F.mse_loss(prediction, target) * ( self.config.mse_lambda or 1)
            self.add_metric('mse', loss)
            return loss, loss

        def off_diagonal(a):
            n = a.shape[1]
            return a.masked_select(~torch.eye(n, dtype=bool, device=a.device))

        self.augs = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
            K.RandomPerspective(0.7,p=0.7),
        )

        def augment(x):
            return self.augs(x)

        def vicreg_loss_image():
            x_a = augment(self.x)
            x_b = augment(self.x)
            z_a = self.encode_image(x_a)
            z_b = self.encode_image(x_b)
            z_a = self.project_image(z_a)
            z_b = self.project_image(z_b)
            D = z_a.shape[1]
            N = z_a.shape[0]
            sim_loss = F.mse_loss(z_a, z_b)
            std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
            std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
            std_loss = torch.mean(torch.relu(1-std_z_a)+torch.mean(torch.relu(1-std_z_b)))
            z_a = z_a - z_a.mean(dim=0)
            z_b = z_b - z_b.mean(dim=0)
            cov_z_a = (z_a.T @ z_a) / (N - 1)
            cov_z_b = (z_b.T @ z_b) / (N - 1)
            cov_loss = off_diagonal(cov_z_a).pow_(2).sum() / D + \
                off_diagonal(cov_z_b).pow_(2).sum() / D

            mu = (self.config.vicreg_mu or 2.5)
            lam = (self.config.vicreg_lambda or 2.5)
            nu = (self.config.vicreg_nu or 1)
            loss = lam * sim_loss + mu * std_loss + nu * cov_loss
            self.add_metric('vicr', loss.mean())
            return loss ,loss

        def random_swap(text, offset, length):
            rot = random.randint(offset,length)
            rot2 = rot
            while rot2 == rot:
                rot2 = random.randint(offset,length)
            newtext = text[:rot] + text[rot2:rot2+1] + text[rot+1:]
            return newtext

        def vicreg_loss_text():
            newtext_a = []
            newtext_b = []
            for text in self.text:
                max_len = len(text)
                max_len = max(max_len, 64)
                newtext_a.append(random_swap(text, 0, max_len//2))
                newtext_b.append(random_swap(text, max_len//2, max_len))
            z_a = self.encode_text(newtext_a)
            z_b = self.encode_text(newtext_b)
            z_a = self.project_text(z_a)
            z_b = self.project_text(z_b)
            D = z_a.shape[1]
            N = z_a.shape[0]
            sim_loss = F.mse_loss(z_a, z_b)
            std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
            std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
            std_loss = torch.mean(torch.relu(1-std_z_a)+torch.mean(torch.relu(1-std_z_b)))
            z_a = z_a - z_a.mean(dim=0)
            z_b = z_b - z_b.mean(dim=0)
            cov_z_a = (z_a.T @ z_a) / (N - 1)
            cov_z_b = (z_b.T @ z_b) / (N - 1)
            cov_loss = off_diagonal(cov_z_a).pow_(2).sum() / D + \
                off_diagonal(cov_z_b).pow_(2).sum() / D

            mu = (self.config.vicreg_mu or 2.5)
            lam = (self.config.vicreg_lambda or 2.5)
            nu = (self.config.vicreg_nu or 1)
            loss = lam * sim_loss + mu * std_loss + nu * cov_loss
            self.add_metric('vict', loss.mean())
            return loss ,loss

        self.add_loss(vicreg_loss_image)
        self.add_loss(vicreg_loss_text)
        self.add_loss(l2_loss)
        self.text_encoder = T5TextEncoder(max_text_len=(self.config.tokens or 64))

    def forward_discriminator(self, *inputs):
        return self.discriminator(*inputs)

    def next_inputs(self):
        inp = self.inputs.next()
        self.x = inp['img']
        self.text = inp['txt']
        self.augmented_latent = self.train_hooks.augment_latent(self.latent.next())
        self.augmented_x = self.train_hooks.augment_x(self.x)

    def encode_image(self, img):
        return self.custom_encoder(img)

    def encode_text(self, text):
        text_embeddings = self.text_encoder.encode_text(text, tokens=(self.config.tokens or 64)).to(self.device)
        return self.custom_text_encoder(text_embeddings)

    def partial_noise(self, x, threshold=0.125):
        mask = (torch.rand_like(x) > threshold).float()
        return mask*x

    def forward_pass(self):
        prompt = self.config.prompt

        if self.config.denoising:
            self.s0 = self.encode_image(self.partial_noise(self.x))
        else:
            self.s0 = self.encode_image(self.x)

        self.encoded_text = self.encode_text(self.text)
        self.s1 = self.pred(torch.cat([self.encoded_text.unsqueeze(1), self.augmented_latent.unsqueeze(1)], 1))
        self.autoencoding = self.generator(self.s0.clone().detach())
        self.st = self.encode_image(self.x)
        self.g_args = [ self.encoded_text.unsqueeze(1), self.s1]
        self.x_args = [ self.encoded_text.unsqueeze(1), self.st.unsqueeze(1)]
        self.g_args = [torch.cat(self.g_args, dim=1)]
        self.x_args = [torch.cat(self.x_args, dim=1).clone().detach()]
        d_real = self.forward_discriminator(*self.x_args)
        d_fake = self.forward_discriminator(*self.g_args)
        self.text_prediction = self.discriminator.context['text_prediction']

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
        return [self.generator, self.pred, self.custom_encoder, self.custom_text_encoder, self.project_image, self.project_text]

    def discriminator_fake_inputs(self):
        return [self.g_args]

    def discriminator_real_inputs(self):
        return self.x_args

