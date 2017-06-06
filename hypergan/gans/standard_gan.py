import importlib
import json
import numpy as np
import os
import sys
import time
import uuid
import copy

from hypergan.discriminators import *
from hypergan.encoders import *
from hypergan.generators import *
from hypergan.loaders import *
from hypergan.samplers import *
from hypergan.trainers import *

import hyperchamber as hc
from hyperchamber import Config
from hypergan.ops import TensorflowOps
import tensorflow as tf
import hypergan as hg

from hypergan.gan_component import ValidationException, GANComponent

class StandardGAN(GANComponent):
    """ GANs (Generative Adversarial Networks) consist of a generator and discriminator(s)."""
    def __init__(self, config=None, graph={}, device='/cpu:0', ops_config=None, ops_backend=TensorflowOps):
        """ Initialized a new GAN."""
        self.device = device
        self.ops_backend = ops_backend
        self.ops_config = ops_config
        self.created = False
        self.components = []

        if config == None:
            config = hg.Configuration.default()

        # A GAN as a component has a parent of itself
        # gan.gan.gan.gan.gan.gan
        GANComponent.__init__(self, self, config)

        self.graph = Config(graph)
        self.inputs = [graph[k] for k in graph.keys()]

    def required(self):
        return "generator".split()

    def sample_input(self):
        #TODO
        return self.ops.session.run(tf.concat(axis=0, values=self.inputs))

    def batch_size(self):
        #TODO how does this work with generators outside of discriminators?
        if len(self.inputs) == 0:
            raise ValidationException("gan.batch_size() requested but no inputs provided")
        return self.ops.shape(self.inputs[0])[0]

    def channels(self):
        #TODO same issue with batch_size
        if len(self.inputs) == 0:
            raise ValidationException("gan.channels() requested but no inputs provided")
        return self.ops.shape(self.inputs[0])[-1]

    def width(self):
        #TODO same issue with batch_size
        if len(self.inputs) == 0:
            raise ValidationException("gan.width() requested but no inputs provided")
        print("----", self.ops.shape(self.inputs[0]))
        return self.ops.shape(self.inputs[0])[2]

    def height(self):
        #TODO same issue with batch_size
        if len(self.inputs) == 0:
            raise ValidationException("gan.height() requested but no inputs provided")
        return self.ops.shape(self.inputs[0])[1]

    def get_config_value(self, symbol):
        if symbol in self.config:
            config = hc.Config(hc.lookup_functions(self.config[symbol]))
            return config
        return None

    def discriminator_variables(self):
        #TODO test
        return self.discriminators[0].ops.variables()

    def generator_variables(self):
        #TODO test
        return self.generator.ops.variables()

    def encoder_variables(self):
        #TODO test
        return self.encoders[0].ops.variables()

    def encoder_z(self):
        #TODO test
        return self.encoders[0].z

    def create(self):
        with tf.device(self.device):
            if self.created:
                print("gan.create already called. Cowardly refusing to create graph twice")
                return

            self.session = self.ops.new_session(self.ops_config)

            config = self.config

            self.encoders = [self.create_component(encoder) for encoder in config.encoders]
            self.generator = self.create_component(config.generator)
            self.discriminators = [self.create_component(discriminator) for discriminator in config.discriminators]
            self.losses = [self.create_component(loss) for loss in config.losses]
            self.trainer = self.create_component(config.trainer)
            self.sampler = self.create_component(config.sampler)
            self.created = True

            self.session.run(tf.global_variables_initializer())

    def create_component(self, defn):
        if defn == None:
            return None
        if defn['class'] == None:
            raise ValidationException("Component definition is missing '" + name + "'")
        gan_component = defn['class'](self, defn)
        gan_component.create()
        self.components.append(gan_component)
        return gan_component

    def step(self, feed_dict={}):
        if not self.created:
            self.create()
        if self.trainer == None:
            raise ValidationException("gan.trainer is missing.  Cannot train.")
        return self.trainer.step(feed_dict)
