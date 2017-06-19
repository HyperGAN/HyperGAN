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
from .standard_gan import StandardGAN
from .base_gan import BaseGAN

class AutoencoderGAN(StandardGAN):
    """ 
    """

    def required(self):
        return "generator".split()

    def create(self):
        config = self.config

        d2 = dict(config.discriminator)
        #d2['class'] = self.ops.lookup("class:hypergan.discriminators.pyramid_discriminator.PyramidDiscriminator")
        self.encoder = self.create_component(d2)
        self.encoder.ops.describe("encoder")
        self.encoder.create(self.inputs.x)
        self.encoder.z = tf.zeros(0)
        self.trainer = self.create_component(config.trainer)


        StandardGAN.create(self)
        cycloss = tf.reduce_mean(tf.abs(self.inputs.x-self.generator.sample))
        cycloss_lambda = config.cycloss_lambda or 10
        self.loss.sample[1] *= config.g_lambda or 1
        self.loss.sample[1] += cycloss*cycloss_lambda

        if config.iterate_zs:
            gsample=self.generator.sample
            dsample=self.discriminator.sample
            for i in range(config.iterate_zs):
                if config.iterate_random_zs:
                    fade = tf.random_uniform([self.gan.batch_size(), 1], 0, 1)
                else:
                    fade=0.5
                z2 = (1-fade)*self.encoder.sample + self.shuffle_batch(self.encoder.sample)*fade
                g2 = self.generator.reuse(z2)
                self.generator.sample = g2 # TODO hidden state variables
                d2 = self.discriminator.reuse(x=self.gan.inputs.x, g=g2)
                self.discriminator.sample = d2
                loss2 = self.create_component(config.loss, discriminator = self.discriminator)
                loss2.create()
                self.loss.sample[0] += loss2.sample[0]
                self.loss.sample[1] += loss2.sample[1]
            self.generator.sample = gsample
            self.discriminator.sample = dsample

        self.trainer.create()

        self.session.run(tf.global_variables_initializer())

    def shuffle_batch(self, net):
        shape = self.ops.shape(net)
        slices = tf.split(net,shape[0])
        print(slices)
        random.shuffle(slices)
        result = tf.stack(slices)
        result = tf.reshape(result, shape)
        print(result)
        return result


