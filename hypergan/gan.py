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

from hypergan.gan_component import ValidationException

class GAN:
    """ GANs (Generative Adversarial Networks) consist of a generator and discriminator(s)."""
    def __init__(self, config, graph, ops=TensorflowOps, device='/gpu:0', graph_type='full', tfconfig=None):
        """ Initialized a new GAN."""
        self.config=Config(config)
        self.device=device
        self.ops = ops
        if tfconfig is None:
            tfconfig = tf.ConfigProto()
            tfconfig.gpu_options.allow_growth=True
        self.init_session(device, tfconfig)
        self.graph = Config(graph)

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

    def init_session(self, device, tfconfig):
        # Initialize tensorflow
        with tf.device(device):
            self.sess = tf.Session(config=tfconfig)

    def train(self, feed_dict={}):
        trainer = self.get_config_value('trainer') 
        if trainer is None:
            raise ValidationException("GAN.train called but no trainer defined")
        return trainer.run(self, feed_dict)
