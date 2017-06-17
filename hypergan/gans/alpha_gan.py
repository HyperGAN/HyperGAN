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
from hypergan.trainers.alpha_trainer import AlphaTrainer

class AlphaGAN(BaseGAN):
    """ 
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
        return "generator".split()

    def create(self):
        BaseGAN.create(self)
        if self.session is None: 
            self.session = self.ops.new_session(self.ops_config)
        with tf.device(self.device):
            config = self.config
            ops = self.ops

            d2 = dict(config.discriminator)
            #d2['class'] = self.ops.lookup("class:hypergan.discriminators.pyramid_discriminator.PyramidDiscriminator")
            encoder = self.create_component(d2)
            encoder.ops.describe("encoder")
            encoder.create(self.inputs.x)
            encoder.z = tf.zeros(0)

            encoder_discriminator = FullyConnectedDiscriminator(self, {})
            encoder_discriminator.ops.describe("encoder_discriminator")
            standard_discriminator = self.create_component(config.discriminator)
            standard_discriminator.ops.describe("discriminator")

            uniform_encoder_config = config.encoder
            uniform_encoder_config.z = ops.shape(encoder.sample)[1]//len(uniform_encoder_config.projections)
            uniform_encoder = UniformEncoder(self, uniform_encoder_config)
            uniform_encoder.create()

            self.generator = self.create_component(config.generator)

            z = uniform_encoder.sample
            x = self.inputs.x
            z_hat = encoder.sample
            g = self.generator.create(z)
            x_hat = self.generator.reuse(z_hat)

            encoder_discriminator.create(x=z, g=z_hat)

            encoder_loss = self.create_component(config.loss, discriminator = encoder_discriminator)
            encoder_loss.create()

            stacked_xg = ops.concat([x, x_hat, g], axis=0)
            standard_discriminator.create(stacked_xg)

            standard_loss = self.create_component(config.loss, discriminator = standard_discriminator)
            standard_loss.create()

            self.trainer = self.create_component(config.trainer)

            #loss terms
            cycloss = tf.reduce_mean(tf.abs(self.inputs.x-x_hat))
            cycloss_lambda = config.cycloss_lambda or 10
            loss1=cycloss + encoder_loss.g_loss
            loss2=cycloss + standard_loss.g_loss
            loss3=standard_loss.d_loss
            loss4=encoder_loss.d_loss

            var_lists = []
            var_lists.append(encoder.variables())
            var_lists.append(self.generator.variables())
            var_lists.append(standard_discriminator.variables())
            var_lists.append(encoder_discriminator.variables())

            # trainer

            self.trainer = AlphaTrainer(self, self.config.trainer, [loss1,loss2,loss3,loss4], var_lists=var_lists)
            self.trainer.create()

            self.session.run(tf.global_variables_initializer())

            self.encoder = encoder


    def step(self, feed_dict={}):
        return self.trainer.step(feed_dict)