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

from hypergan.encoders.uniform_encoder import UniformEncoder
from hypergan.trainers.multi_step_trainer import MultiStepTrainer

class MultiGeneratorGAN(BaseGAN):
    """ 
    https://arxiv.org/pdf/1708.02556v2.pdf
    """
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)
        self.discriminator = None
        self.encoder = None
        self.generator = None
        self.loss = None
        self.trainer = None
        self.session = None

    def required(self):
        return "generator discriminator number_generators".split()

    def create(self):
        BaseGAN.create(self)
        if self.session is None: 
            self.session = self.ops.new_session(self.ops_config)
        with tf.device(self.device):
            config = self.config
            print(config)
            print("________")
            ops = self.ops


            self.encoder = self.create_component(config.encoder)
            self.encoder.create()
            generator_samples = []
            config.generator.skip_linear = True

            print("!!!!!!!!!!!!!!!!!!!!!!! Creataing generator", config.generator)
            generator = self.create_component(config.generator)
            generator.ops.describe("generator")
            self.generator = generator
            for i in range(config.number_generators):
                primes = config.generator.initial_dimensions or [4, 4]
                initial_depth = generator.depths(primes[0])[0]
                net = ops.reshape(self.encoder.sample, [ops.shape(self.encoder.sample)[0], -1])
                new_shape = [ops.shape(net)[0], primes[0], primes[1], initial_depth]
                net = ops.linear(net, initial_depth*primes[0]*primes[1])
                pi = ops.reshape(net, new_shape)
         
                #pi = tf.zeros([self.batch_size(), primes[0], primes[1], 256])
                print("[MultiGeneratorGAN] Creating generator ", i, pi)
                if i == 0:
                    gi = generator.create(pi)
                else:
                    gi = generator.reuse(pi)
                generator_samples.append(gi)
 
            self.discriminator = self.create_component(config.discriminator)
            self.discriminator.ops.describe("discriminator")

            losses = []
            self.loss = self

            self.loss = self.create_component(config.loss)

            g_loss = tf.constant(0.0)
            d_loss = tf.constant(0.0)
            metrics = [None, None]
            d_fake_features = []

            for i in range(config.number_generators):
                if i == 0:
                    di = self.discriminator.create(x=self.inputs.x, g=generator_samples[i])
                else:
                    di = self.discriminator.reuse(x=self.inputs.x, g=generator_samples[i])
                d_real, d_fake = self.split_batch(di, 2)
                # after the divergence measure or before ? TODO
                #d_fake_features.append(self.discriminator.g_loss_features)
                d_fake_features.append(d_fake)

                loss = self.loss.create(d_real=d_real, d_fake=d_fake)
                losses.append(loss)
                g_loss += loss[1]
                d_loss += loss[0]

            if config.class_loss_type == 'svm':
                # classifier loss 
                for i in range(config.number_generators):
                    features = tf.reshape(d_fake_features[i], [self.batch_size(), -1])
                    c_loss = ops.lookup('tanh')(features)
                    c_loss = ops.linear(c_loss, config.number_generators)
                    label = tf.one_hot([i], config.number_generators)
                    label = tf.tile(label, [self.batch_size(), 1])
                    c_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=c_loss, labels=label)
            
                    g_loss += c_loss*(config.c_loss_lambda or 1)

            var_lists = []
            var_lists.append(self.generator.variables() + self.variables())
            var_lists.append(self.discriminator.variables())

            # trainer
            steps = [('generator', g_loss), ('discriminator', d_loss)]


            print("T", self.config.trainer, steps, metrics)
            self.trainer = MultiStepTrainer(self, self.config.trainer, steps, var_lists=var_lists, metrics=metrics)
            self.trainer.create()

            self.session.run(tf.global_variables_initializer())
            self.uniform_sample = tf.concat(generator_samples, axis=1)

