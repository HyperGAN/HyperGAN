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
from .standard_gan import StandardGAN

class DistributionFilteringGAN(StandardGAN):
    """
    On Stabilizing Generative Adversarial Training with Noise
    https://arxiv.org/pdf/1906.04612v1.pdf
    """
    def create(self):
        config = self.config

        with tf.device(self.device):
            self.session = self.ops.new_session(self.ops_config)
            self.latent = self.create_component(config.z_distribution or config.latent)
            self.uniform_distribution = self.latent

            z_shape = self.ops.shape(self.latent.sample)
            self.android_input = tf.reshape(self.latent.sample, [-1])

            direction, slider = self.create_controls(self.ops.shape(self.android_input))
            self.slider = slider
            self.direction = direction
            z = self.android_input + slider * direction
            z = tf.maximum(-1., z)
            z = tf.minimum(1., z)
            z = tf.reshape(z, z_shape)
            self.control_z = z

            self.generator = self.create_component(config.generator, name="generator", input=z)
            self.noise_generator = self.create_component((config.noise_generator or config.generator), name="noise_generator", input=z)

            #x, g = tf.concat([self.inputs.x, self.inputs.x + self.noise_generator.sample], axis=3), tf.concat([self.generator.sample, self.generator.sample + self.noise_generator.sample], axis=3)

            x1, g1 = self.inputs.x, self.generator.sample
            self.discriminator = self.create_component(config.discriminator, name="discriminator", input=tf.concat([x1,g1],axis=0))
            x2, g2 = self.inputs.x+self.noise_generator.sample, self.generator.sample+self.noise_generator.sample
            self.loss = self.create_component(config.loss, discriminator=self.discriminator)
            self.noise_discriminator = self.create_component(config.discriminator, name="discriminator", input=tf.concat([x2,g2],axis=0), reuse=True)
            noise_loss = self.create_component(config.loss, discriminator=self.noise_discriminator)
            self.loss.sample[0] += noise_loss.sample[0]
            self.loss.sample[1] += noise_loss.sample[1]
            self.trainer = self.create_component(config.trainer)

            self.android_output = tf.reshape(self.generator.sample, [-1])

            self.session.run(tf.global_variables_initializer())

    def g_vars(self):
        return self.latent.variables() + self.generator.variables() + self.noise_generator.variables()
    def d_vars(self):
        return self.discriminator.variables()
