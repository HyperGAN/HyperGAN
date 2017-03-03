from hypergan.util.ops import *

from tensorflow.python.framework import ops
from hyperchamber import Config

import copy

import hyperchamber as hc

import hypergan as hg

from hypergan.discriminators import *
from hypergan.encoders import *
from hypergan.generators import *
from hypergan.loaders import *
from hypergan.samplers import *
from hypergan.trainers import *
from hypergan.util import *

import importlib
import json
import numpy as np
import os
import sys
import tensorflow
import tensorflow as tf
import time
import uuid

class GAN:
    """ GANs (Generative Adversarial Networks) consist of a generator and discriminator(s)."""
    def __init__(self, config, graph, device='/gpu:0', graph_type='full'):
        """ Initialized a new GAN."""
        self.config=Config(config)
        self.device=device
        self.init_session(device)
        self.graph = Config(graph)
        #TODO rename me.  Graph should refer to our {name => Tensor} collection
        self.create_graph(graph_type, device)

    def sample_to_file(self, name, sampler=grid_sampler.sample):
        return sampler(self, name)

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

    def init_session(self, device):
        # Initialize tensorflow
        with tf.device(device):
            self.sess = tf.Session(config=tf.ConfigProto())

    def train(self):
        trainer = hc.Config(hc.lookup_functions(self.config.trainer))
        return trainer.run(self)

    def save(self, save_file):
        saver = tf.train.Saver()
        saver.save(self.sess, save_file)

    def load_or_initialize_graph(self, save_file):
        save_file = os.path.expanduser(save_file)
        if os.path.isfile(save_file) or os.path.isfile(save_file + ".index" ):
            print(" |= Loading network from "+ save_file)
            dir = os.path.dirname(save_file)
            print(" |= Loading checkpoint from "+ dir)
            ckpt = tf.train.get_checkpoint_state(os.path.expanduser(dir))
            if ckpt and ckpt.model_checkpoint_path:
                saver = tf.train.Saver()
                saver.restore(self.sess, save_file)
                loadedFromSave = True
                print("Model loaded")
            else:
                print("No checkpoint file found")
        else:
            self.initialize_graph()
    
    def initialize_graph(self):
        print(" |= Initializing new network")
        with tf.device(self.device):
            init = tf.global_variables_initializer()
            self.sess.run(init)
