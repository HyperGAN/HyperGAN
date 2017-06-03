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

class GAN(GANComponent):
    """ GANs (Generative Adversarial Networks) consist of a generator and discriminator(s)."""
    def __init__(self, config=None, graph={}, device='/gpu:0', ops_config=None, ops_backend=TensorflowOps):
        """ Initialized a new GAN."""
        if config == None:
            config = hg.Configuration.default()
        self.ops_backend = ops_backend
        GANComponent.__init__(self, self, config)
        self.session = self.ops.new_session(device, ops_config)
        self.graph = Config(graph)

    def required(self):
        return "generator".split()

    def sample_to_file(self, name, sampler=static_batch_sampler.sample):
        return sampler(self, name)

    def get_config_value(self, symbol):
        if symbol in self.config:
            config = hc.Config(hc.lookup_functions(self.config[symbol]))
            return config
        return None

    def create_graph(self, graph_type, device):
        tf_graph = hg.graph.Graph(self)
        graph = self.graph
        with tf.device(device):
            if 'y' in graph:
                # convert to one-hot
                graph.y=tf.cast(graph.y,tf.int64)
                graph.y=tf.one_hot(graph.y, self.config['y_dims'], 1.0, 0.0)

            if graph_type == 'full':
                tf_graph.create(graph)
            elif graph_type == 'generator':
                tf_graph.create_generator(graph)
            else:
                raise Exception("Invalid graph type")

    def train(self, feed_dict={}):
        trainer = self.get_config_value('trainer') 
        if trainer is None:
            raise ValidationException("GAN.train called but no trainer defined")
        return trainer.run(self, feed_dict)
