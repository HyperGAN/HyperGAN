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
        self.generators_cache = {}
        self.discriminators_cache = {}
        self.encoders_cache = {}
        self.latents_cache = {}
        self._d_vars = []
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

    def zN(self, zi):
        reuse = False
        name = "z%d" % (int(zi))

        if name in self.latents_cache:
            reuse = True
        latent = self.create_component(self.config.latent, name=name, reuse=reuse)
        self.latents_cache[name] = latent
        return latent.sample


    def geN(self, gi, ei, xi, encoder=None):
        if encoder is None:
            encoder = self.eN(ei, xi)
        reuse = False
        name = "g%de%dx%d" % (int(gi), int(ei), int(xi))
        if name in self.generators_cache:
            reuse = True
        generator = self.create_component(self.config.generator, name=name, input=encoder, reuse=reuse)
        self.generators_cache[name] = generator
        return generator.sample

    def gegeN(self, gi2, ei2, gi, ei, xi):
        g1 = self.geN(gi, ei, xi)
        name = "e%dg%de%dx%d" % (int(ei2), int(gi2), int(ei), int(xi))
        reuse=False
        if name in self.encoders_cache:
            reuse = True
        e2 = self.create_component(self.config.encoder, name=name, input=g1, reuse=reuse)
        self.encoders_cache[name] = e2

        reuse = False
        name = "g%de%dg%de%dx%d" % (int(gi2), int(ei2), int(gi), int(ei), int(xi))
        if name in self.generators_cache:
            reuse = True
        generator = self.create_component(self.config.generator, name=name, input=e2.sample, reuse=reuse)
        self.generators_cache[name] = generator
        return generator.sample

    def gzN(self, gi, zi):
        z = self.zN(zi)
        reuse = False
        name = "g%dz%d" % (int(gi), int(zi))
        if name in self.generators_cache:
            reuse = True
        generator = self.create_component(self.config.generator, name=name, input=z, reuse=reuse)
        self.generators_cache[name] = generator
        self.generator = generator
        return generator.sample

    def parse_opts(self, opts):
        options = {}
        for opt in opts.split(","):
            if opt == "":
                continue
            name, val = opt.split("=")
            value = self.configurable_param(val)
            options[name]=value
        return hc.Config(options)

    def adversarialxN(self, xi, opts):
        options = self.parse_opts(opts)
        x = self.xN(xi)
        return self.adversarial("x"+str(xi), x, options)

    def adversarialgzN(self, gi, zi, opts):
        options = self.parse_opts(opts)
        g = self.gzN(gi, zi)
        return self.adversarial("g"+str(gi)+"z"+str(zi), g, options)

    def adversarialgeN(self, gi, ei, xi, opts):
        options = self.parse_opts(opts)
        g = self.geN(gi, ei, xi)
        return self.adversarial("g"+str(gi)+"e"+str(ei)+"x"+str(xi), g, options)

    def adversarial(self, name, src, options):
        trainable=tf.Variable(tf.zeros_like(self.inputs.x), name='adversarial'+name)
        self._d_vars += [trainable]
        clear=tf.assign(trainable, tf.zeros_like(src))
        ops = self.ops
        with tf.get_default_graph().control_dependencies([clear]):
            din = tf.concat([(src+trainable), (src+trainable)], axis=0)
            dN = 0
            reuse = False
            if dN in self.discriminators_cache:
                reuse = True
            tfname = "d"+str(dN)
            adversarial_discriminator = self.create_component(self.config.discriminator, name=tfname, input=din, reuse=reuse)
            self.discriminators_cache[dN] = adversarial_discriminator
            loss = self.create_component(self.config.loss, discriminator=adversarial_discriminator)
            v = tf.gradients(adversarial_discriminator.sample, trainable)[0]
            v = tf.stop_gradient(v)
            if self.config.adversarial_calculation == "":
                adv = v/tf.norm(v, ord=2)
            else:
                adv = tf.sign(v)

            return src + (options.gamma or self.config.adversarial_gamma or (1/255.0))*adv

    def mixgzNxN(self, xi, gi):
        pass

    def required(self):
        return "terms".split()

    def resolve_term(self, term):
        matching = {
                "gN(eN(xN))": self.geN,
                "gN(eN(gN(eN(xN))))": self.gegeN,
                "gN(zN)": self.gzN,
                "adversarial(xN,options)": self.adversarialxN,
                "adversarial(gN(zN),options)": self.adversarialgzN,
                "adversarial(gN(eN(xN)),options)": self.adversarialgeN,
                "mix(g(zN)xN,options)": self.mixgzNxN,
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
                terms += [self.resolve_term(dt)]
            reuse = False

            if dN == "l2":
                loss = hc.Config({"sample": tf.square(terms[0]-terms[1])})
            if dN[0:3] == "l2d":
                dN = dN.replace("l2d", "")
                dN = re.findall("\d+", dN)[0]
                dN = int(dN)
                tfname = "d"+str(dN)
                d0 = self.create_component(config.discriminator, name=tfname, input=tf.concat(terms[0:2],axis=0), reuse=True)
                d1 = self.create_component(config.discriminator, name=tfname, input=tf.concat(terms[2:4],axis=0), reuse=True)
                flex = self.config.l2d_flex or 0.0
                loss = hc.Config({"sample": tf.nn.relu(tf.abs(d1.sample-d0.sample) - flex)})
            else:
                dN = re.findall("\d+", dN)[0]
                dN = int(dN)
                if dN in self.discriminators_cache:
                    reuse = True
                tfname = "d"+str(dN)
                d = self.create_component(config.discriminator, name=tfname, input=tf.concat(terms,axis=0), reuse=reuse)
                self.discriminators_cache[dN] = d
                loss = self.create_component(config.loss, discriminator=d, split=len(d_terms))

            if config.term_gammas:
                loss.sample = [self.configurable_param(config.term_gammas[i]) * loss.sample[0], self.configurable_param(config.term_gammas[i]) * loss.sample[1]]
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

    def variables(self):
        return super().variables() + self._d_vars
