import importlib
import json
import numpy as np
import os
import sys
import tensorflow
import tensorflow as tf
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
import hypergan as hg

class GAN:
    """ GANs (Generative Adversarial Networks) consist of a generator and discriminator(s)."""
    def __init__(self, config={}, graph={}, ops=None, generator_only=False):
        """ Initialized a new GAN."""
        self.config = Config(config)
        self.graph = Config(graph)

    def sample_to_file(self, name, sampler=static_batch_sampler.sample):
        return sampler(self, name)

    def get_config_value(self, symbol):
        if symbol in self.config:
            config = hc.Config(hc.lookup_functions(self.config[symbol]))
            return config
        return None

    def train(self, feed_dict={}):
        trainer = self.get_config_value('trainer') 
        if trainer is None:
            raise Exception("GAN.train called but no trainer defined")
        return trainer.run(self, feed_dict)
