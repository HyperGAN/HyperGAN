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
    def required(self):
        return "generator".split()

    def create(self):
        super(StandardGAN, self).create()

        with tf.device(self.device):
            self.session = self.ops.new_session(self.ops_config)

            config = self.config

            self.encoder = self.create_component(config.encoder)
            self.generator = self.create_component(config.generator)
            self.discriminator = self.create_component(config.discriminator)
            self.loss = self.create_component(config.loss)
            self.trainer = self.create_component(config.trainer)

            self.session.run(tf.global_variables_initializer())

    def step(self, feed_dict={}):
        if not self.created:
            self.create()
        if self.trainer == None:
            raise ValidationException("gan.trainer is missing.  Cannot train.")
        return self.trainer.step(feed_dict)
