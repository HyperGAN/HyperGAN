import importlib
import json
import numpy as np
import os
import re
import sys
import time
import uuid
import copy

from hypergan.discriminators import *
from hypergan.distributions import *
from hypergan.generators import *
from hypergan.inputs import *
from hypergan.samplers import *
from hypergan.trainers import *

import hyperchamber as hc
from hyperchamber import Config
from hypergan.ops import TensorflowOps
import hypergan as hg

from hypergan.gan_component import ValidationException, GANComponent
from .base_gan import BaseGAN

class ConfigurableGAN(BaseGAN):
    def __init__(self, *args, **kwargs):
        self.d_terms = []
        self.Ds = []
        BaseGAN.__init__(self, *args, **kwargs)

    def create_encoder(self):
        return self.create_component(self.config.encoder)

    def create_latent(self, zi):
        return self.create_component(self.config.latent)

    def create_generator(self):
        return self.create_component(self.config.generator)

    def parse_opts(self, opts):
        options = {}
        for opt in opts.split(","):
            if opt == "":
                continue
            name, val = opt.split("=")
            value = self.configurable_param(val)
            options[name]=value
        return hc.Config(options)

    def required(self):
        return "terms".split()

    def create_term(self, term):
        for match in 
        matching = {
                "gN(eN(xN))": self.create_generator,
                "gN(zN)": self.create_generator
                }
        for regex, method in matching.items():
            regex_subbed = regex.replace("(", '\(').replace(")", '\)').replace("N", "(\d+)?").replace(",options", "([-,=\w\d\.\(\)]+)?")
            regex = re.compile(regex_subbed)
            args = re.match(regex, term)
            if args:
                return method(*args.groups())

        raise ValidationException("Could not match term: " + term)

    def forward_term(self, term):
        matching = {
                "gN(eN(xN))": self.geN,
                "gN(zN)": self.gzN,
                "xN": self.xN
                }
        for regex, method in matching.items():
            regex_subbed = regex.replace("(", '\(').replace(")", '\)').replace("N", "(\d+)?").replace(",options", "([-,=\w\d\.\(\)]+)?")
            regex = re.compile(regex_subbed)
            args = re.match(regex, term)
            if args:
                return method(*args.groups())

        raise ValidationException("Could not match term: " + term)

    def create(self):
        config = self.config

        self.latent = hc.Config({"sample": self.zN(0)})
        self.discriminators = []
        self.losses = []
        for i,term in enumerate(self.config.terms):
            dN, args = term.split(":")

            d_terms = args.split("/")
            terms = []
            for dt in d_terms:
                terms += (term,self.create_term(dt))
            reuse = False

            dN = re.findall("\d+", dN)[0]
            dN = int(dN)
            tfname = "d"+str(dN)
            D = self.create_component(config.discriminator)
            self.Ds.append(D)
            self.d_terms += terms
 
        self.trainer = self.create_component(config.trainer)

    def create_controls(self, z_shape):
        direction = tf.constant(0.0, shape=z_shape, name='direction') * 1.00
        slider = tf.constant(0.0, name='slider', dtype=tf.float32) * 1.00
        return direction, slider

    def forward_pass(self):
        d_reals = []
        d_fakes = []
        for terms in d_terms:

        return d_reals, d_fakes

    def forward_loss(self):
        losses = []
        for d_real, d_fake in zip(d_reals, d_fakes):
            loss = self.create_component(config.loss, discriminator=d, split=len(d_terms))
            d_loss, g_loss = loss.forward(d_real, d_fake)
            d_loss = [self.configurable_param(config.term_gammas[i]) * d_loss, self.configurable_param(config.term_gammas[i]) * g_loss]
            losses += [[d_loss, g_loss]]

        self.loss = hc.Config({
            'sample': [sum([l.sample[0] for l in losses]), sum([l.sample[1] for l in losses])]
            })

    def regularize_adversarial_norm(self):
        x = Variable(self.x, requires_grad=True).cuda()
        d1_logits = self.discriminator(x)
        d2_logits = self.d_fake

        loss = loss.forward_adversarial_norm(d1_logits, d2_logits)

        if loss == 0:
            return [None, None]

        d1_grads = torch_grad(outputs=loss, inputs=x, retain_graph=True, create_graph=True)
        d1_norm = [torch.norm(_d1_grads.view(-1).cuda(),p=2,dim=0) for _d1_grads in d1_grads]

        reg_d1 = [((_d1_norm**2).cuda()) for _d1_norm in d1_norm]
        reg_d1 = sum(reg_d1)

        return loss, reg_d1

    def g_parameters(self):
        params = []
        for d_terms in self.d_terms:
            for term in d_terms:
                params += term[1].parameters()
        return params

    def d_parameters(self):
        params = []
        for m in self.Ds:
            params += m.parameters()
        return params
