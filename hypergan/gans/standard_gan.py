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

class StandardGAN(BaseGAN):
    """ 
    Standard GANs consist of:
    
    *required to sample*
    
    * encoder
    * generator
    * sampler

    *required to train*

    * discriminator
    * loss
    * trainer
    """
    def __init__(self, *args, **kwargs):
        self.discriminator = None
        self.encoder = None
        self.generator = None
        self.loss = None
        self.trainer = None
        self.session = None
        BaseGAN.__init__(self, *args, **kwargs)

    def required(self):
        return "generator".split()

    def create(self):
        config = self.config

        with tf.device(self.device):
            if self.session is None: 
                self.session = self.ops.new_session(self.ops_config)

            #this is in a specific order
            if self.encoder is None and config.encoder:
                self.encoder = self.create_component(config.encoder)
                self.uniform_encoder = self.encoder
            if self.generator is None and config.generator:
                self.generator = self.create_component(config.generator)
                self.uniform_sample = self.generator.sample

            if self.discriminator is None and config.discriminator:
                self.discriminator = self.create_component(config.discriminator)
                self.discriminator.ops.describe("discriminator")
            if self.loss is None and config.loss:
                self.loss = self.create_component(config.loss)
                self.metrics = self.loss.metrics
            if self.trainer is None and config.trainer:
                self.trainer = self.create_component(config.trainer)


            self.session.run(tf.global_variables_initializer())

