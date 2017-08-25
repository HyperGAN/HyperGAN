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

from hypergan.discriminators.fully_connected_discriminator import FullyConnectedDiscriminator
from hypergan.encoders.uniform_encoder import UniformEncoder
from hypergan.trainers.multi_step_trainer import MultiStepTrainer

class AlphaGAN(BaseGAN):
    """ 
      AlphaGAN, or α-GAN from https://arxiv.org/pdf/1706.04987.pdf
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
        """
        `g_encoder` is a discriminator.  It takes X as input
        `z_discriminator` is another discriminator.  It takes as input the output of g_encoder and z
        `discriminator` is a standard discriminator.  It measures X, reconstruction of X, and G.
        `generator` produces two samples, g_encoder output and a known random distribution.
        """
        return "generator discriminator z_discriminator g_encoder".split()

    def create(self):
        BaseGAN.create(self)
        if self.session is None: 
            self.session = self.ops.new_session(self.ops_config)
        with tf.device(self.device):
            self.inputs.x = tf.identity(self.inputs.x, name='input')
            config = self.config
            ops = self.ops

            g_encoder = dict(config.g_encoder or config.discriminator)
            encoder = self.create_component(g_encoder)
            encoder.ops.describe("g_encoder")
            encoder.create(self.inputs.x)
            encoder.z = tf.zeros(0)
            if(len(encoder.sample.get_shape()) == 2):
                s = ops.shape(encoder.sample)
                encoder.sample = tf.reshape(encoder.sample, [s[0],s[1], 1, 1])

            z_discriminator = dict(config.z_discriminator or config.discriminator)
            z_discriminator['layer_filter']=None

            encoder_discriminator = self.create_component(z_discriminator)
            encoder_discriminator.ops.describe("z_discriminator")
            standard_discriminator = self.create_component(config.discriminator)
            standard_discriminator.ops.describe("discriminator")

            #encoder.sample = ops.reshape(encoder.sample, [ops.shape(encoder.sample)[0], -1])
            uniform_encoder_config = config.encoder
            z_size = 1
            for size in ops.shape(encoder.sample)[1:]:
                z_size *= size
            uniform_encoder_config.z = z_size // len(uniform_encoder_config.projections)
            uniform_encoder = UniformEncoder(self, uniform_encoder_config)
            uniform_encoder.create()

            self.generator = self.create_component(config.generator)

            direction = tf.random_normal(ops.shape(uniform_encoder.sample), stddev=0.3, name='direction')
            slider = tf.get_variable('slider', initializer=tf.constant_initializer(0.0), shape=[1, 1], dtype=tf.float32, trainable=False)
            x = self.inputs.x

            # project the output of the autoencoder
            z_hat = encoder.sample

            z = uniform_encoder.sample + slider * direction
            z = ops.reshape(z, ops.shape(z_hat))
            # end encoding

            g = self.generator.create(z)
            sample = self.generator.sample
            self.uniform_sample = g
            x_hat = self.generator.reuse(z_hat)

            encoder_discriminator.create(x=z, g=z_hat)

            eloss = dict(config.loss)
            eloss['gradient_penalty'] = False
            eloss['gradient_locally_stable'] = False
            encoder_loss = self.create_component(eloss, discriminator = encoder_discriminator)
            encoder_loss.create()

            stacked_xg = ops.concat([x, x_hat, g], axis=0)
            standard_discriminator.create(stacked_xg)

            sloss = dict(config.loss)
            sloss['gradient_penalty'] = False#self.gan.inputs.x
            standard_loss = self.create_component(sloss, discriminator = standard_discriminator)
            standard_loss.create(split=3)

            self.trainer = self.create_component(config.trainer)

            #loss terms
            distance = config.distance or ops.lookup('l1_distance')
            cycloss = tf.reduce_mean(distance(self.inputs.x,x_hat))
            cycloss_lambda = config.cycloss_lambda
            if cycloss_lambda is None:
                cycloss_lambda = 10
            cycloss *= cycloss_lambda
            loss1=('generator', cycloss + encoder_loss.g_loss)
            loss2=('generator', cycloss + standard_loss.g_loss)
            loss3=('discriminator', standard_loss.d_loss)
            loss4=('discriminator', encoder_loss.d_loss)

            var_lists = []
            var_lists.append(encoder.variables())
            var_lists.append(self.generator.variables())
            var_lists.append(standard_discriminator.variables())
            var_lists.append(encoder_discriminator.variables())

            metrics = []
            metrics.append(None)
            metrics.append(None)
            #metrics.append(None)
            
            metrics.append(standard_loss.metrics)
            metrics.append(encoder_loss.metrics)

            # trainer

            self.trainer = MultiStepTrainer(self, self.config.trainer, [loss1,loss2,loss3,loss4], var_lists=var_lists, metrics=metrics)
            self.trainer.create()

            self.session.run(tf.global_variables_initializer())

            self.encoder = encoder
            self.uniform_encoder = uniform_encoder

            self.slider = slider
            self.direction = direction


    def step(self, feed_dict={}):
        return self.trainer.step(feed_dict)

    def input_nodes(self):
        "used in hypergan build"
        #return []
        return [self.inputs.x, self.uniform_encoder.sample, self.slider, self.direction]


    def output_nodes(self):
        "used in hypergan build"
        #return [self.uniform_sample]
        return [self.encoder.sample, self.uniform_sample]
