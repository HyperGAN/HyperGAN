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
import tensorflow as tf
import hypergan as hg

from hypergan.gan_component import ValidationException, GANComponent
from .base_gan import BaseGAN

class ConfigurableGAN(BaseGAN):
    def __init__(self, *args, **kwargs):
        self.generators_cache = {}
        self.discriminators_cache = {}
        self.encoders_cache = {}
        BaseGAN.__init__(self, *args, **kwargs)

    def xN(self, xi):
        return self.inputs.xs[int(xi)]

    def eN(self, ei, xi):
        reuse = False
        name = "e%dx%d" % (int(ei), int(xi))
        if name in self.encoders_cache:
            reuse = True
        encoder = self.create_component(self.config.encoder, name=name, input=self.xN(xi), reuse=reuse)
        self.encoders_cache[name] = encoder
        return encoder.sample

    def geN(self, gi, ei, xi):
        encoder = self.eN(ei, xi)
        reuse = False
        name = "g%de%dx%d" % (int(gi), int(ei), int(xi))
        if name in self.generators_cache:
            reuse = True
        generator = self.create_component(self.config.generator, name=name, input=encoder, reuse=reuse)
        self.generators_cache[name] = generator
        return generator.sample

    def gzN(self, zi):
        pass

    def required(self):
        return "terms".split()

    def resolve_term(self, term):
        matching = {
                "gN(eN(xN))": self.geN,
                "gN(zN)": self.gzN,
                "xN": self.xN
                }
        for regex, method in matching.items():
            regex = re.compile(regex.replace("(", '\(').replace(")", '\)').replace("N", "(\d+)?"))
            args = re.match(regex, term)
            if args:
                return method(*args.groups())

        raise ValidationException("Could not match term: " + term)

    def create(self):
        config = self.config

        if config.z_distribution or config.latent:
            self.latent = self.create_component(config.z_distribution or config.latent)
        self.discriminators = []
        self.losses = []
        for i,term in enumerate(self.config.terms):
            dN, args = term.split(":")
            dN = re.findall("\d+", dN)[0]
            dN = int(dN)
            d_terms = args.split("/")
            terms = []
            for dt in d_terms:
                terms += [self.resolve_term(dt)]
            reuse = False
            if dN in self.discriminators_cache:
                reuse = True
            tfname = "d"+str(dN)
            d = self.create_component(config.discriminator, name=tfname, input=tf.concat(terms,axis=0), reuse=reuse)
            self.discriminators_cache[dN] = d
            loss = self.create_component(config.loss, discriminator=d, split=len(d_terms))
            if config.term_gammas:
                loss.sample = [self.gan.configurable_param(config.term_gammas[i]) * loss.sample[0], self.gan.configurable_param(config.term_gammas[i]) * loss.sample[1]]
            self.losses += [loss]

        self.loss = hc.Config({
            'sample': [sum([l.sample[0] for l in self.losses]), sum([l.sample[1] for l in self.losses])]
            })
 
        self.trainer = self.create_component(config.trainer)

    def create_controls(self, z_shape):
        direction = tf.constant(0.0, shape=z_shape, name='direction') * 1.00
        slider = tf.constant(0.0, name='slider', dtype=tf.float32) * 1.00
        return direction, slider

    def g_vars(self):
        g_vars = []
        for g in self.generators_cache.values():
            g_vars += g.variables()
        for e in self.encoders_cache.values():
            g_vars += e.variables()
        return g_vars

    def d_vars(self):
        d_vars = []
        for d in self.discriminators_cache.values():
            d_vars += d.variables()
        return d_vars
