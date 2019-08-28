import importlib
import json
import numpy as np
import os
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
from ..base_gan import BaseGAN

from hypergan.distributions.uniform_distribution import UniformDistribution

class MultiGeneratorGAN(BaseGAN):
    """ 
    https://arxiv.org/pdf/1708.02556v2.pdf
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
            metrics = []
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

            var_lists = []
            steps = []

            if config.class_loss_type == 'svm':
                # classifier loss 
                for i in range(config.number_generators):
                    features = tf.reshape(d_fake_features[i], [self.batch_size(), -1])
                    c_loss = ops.lookup('crelu')(features)
                    print("C LOSS 1", c_loss)
                    c_loss = ops.linear(c_loss, config.number_generators)
                    label = tf.one_hot([i], config.number_generators)
                    label = tf.tile(label, [self.batch_size(), 1])
                    c_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=c_loss, labels=label)
            
                    g_loss += c_loss*(config.c_loss_lambda or 1)
                metrics.append({"class loss": self.ops.squash(c_loss)})
                metrics.append(self.loss.metrics)


                var_lists.append(self.generator.variables() + self.variables())
                var_lists.append(self.discriminator.variables())
                steps = [('generator', g_loss), ('discriminator', d_loss)]

            if config.class_loss_type == 'gan':
                d2 = None
                g2 = None
                l2 = None
                d2_loss_sum = tf.constant(0.)
                g2_loss_sum = tf.constant(0.)

                var_lists.append(self.generator.variables() + self.variables())
                var_lists.append(self.discriminator.variables())
                # classifier as gan loss 
                for i in range(config.number_generators):
                    if i != 0:
                        self.ops.reuse()
                    features = tf.reshape(d_fake_features[i], [self.batch_size(), -1])
                    label = tf.one_hot([i], config.number_generators)
                    label = tf.tile(label, [self.batch_size(), 1])

                    # D2(G2(gx), label)
                    g2config = dict(config.generator2)
                    g2 = self.create_component(g2config)
                    # this is the generator
                    g2.ops.describe("G2")

                    if i == 0:
                        g2sample = g2.create(tf.concat(generator_samples, axis=3))
                    else:
                        g2sample = g2.reuse(tf.concat(generator_samples, axis=3))


                    d2config = dict(config.discriminator2)
                    d2 = self.create_component(d2config)

                    d2.ops.describe("D2")
                    if i == 0:
                        d2.create(x=label, g=g2sample)
                        var_lists.append(g2.variables())
                        var_lists.append(d2.variables())
                    else:
                        d2.reuse(x=label, g=g2sample)

                    l2config = dict(config.loss)
                    l2 = self.create_component(l2config,discriminator=d2, generator=g2)
                    d2_loss, g2_loss = l2.create()

                    g2_loss_sum += g2_loss
                    d2_loss_sum += d2_loss

                    if i != 0:
                        self.ops.stop_reuse()


                steps = [
                        ('generator 1', g_loss + g2_loss_sum), 
                        ('discriminator 1', d_loss),
                        ('generator 2', g2_loss_sum + g_loss), 
                        ('discriminator 2', d2_loss)
                ]

                metrics.append(None)
                metrics.append(self.loss.metrics)
                metrics.append(None)
                metrics.append(l2.metrics)



            print("T", self.config.trainer, steps, metrics)
            self.trainer = MultiStepTrainer(self, self.config.trainer, steps, var_lists=var_lists, metrics=metrics)
            self.trainer.create()

            self.session.run(tf.global_variables_initializer())
            self.uniform_sample = tf.concat(generator_samples, axis=1)

