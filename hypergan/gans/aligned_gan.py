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
from hypergan.trainers.multi_step_trainer import MultiStepTrainer
from .base_gan import BaseGAN
from .standard_gan import StandardGAN

class AlignedGAN(BaseGAN):
    def required(self):
        return ["generator", "discriminator"]

    def create(self):
        config = self.config
        ops = self.ops

        self.session = self.ops.new_session(self.ops_config)

        encoder_config = dict(config.input_encoder)
        encode_a = self.create_component(encoder_config)
        encode_a.ops.describe("encode_a")
        encode_b = self.create_component(encoder_config)
        encode_b.ops.describe("encode_b")

        g_ab = self.create_component(config.generator)
        g_ab.ops.describe("g_ab")
        g_ba = self.create_component(config.generator)
        g_ba.ops.describe("g_ba")

        #encode_a.ops = g_ab.ops
        #encode_b.ops = g_ba.ops

        encode_a.create(self.inputs.xa)
        encode_b.create(self.inputs.xb)

        g_ab.create(encode_a.sample)
        g_ba.create(encode_b.sample)

        self.xba = g_ba.sample
        self.xab = g_ab.sample

        discriminator_a = self.create_component(config.discriminator)
        discriminator_b = self.create_component(config.discriminator)
        discriminator_a.ops.describe("discriminator_a")
        discriminator_b.ops.describe("discriminator_b")
        discriminator_a.create(x=self.inputs.xa, g=g_ba.sample)
        discriminator_b.create(x=self.inputs.xb, g=g_ab.sample)

        encode_g_ab = encode_b.reuse(g_ab.sample)
        encode_g_ba = encode_a.reuse(g_ba.sample)

        cyca = g_ba.reuse(encode_g_ab)
        cycb = g_ab.reuse(encode_g_ba)

        lossa = self.create_component(config.loss, discriminator=discriminator_a, generator=g_ba)
        lossb = self.create_component(config.loss, discriminator=discriminator_b, generator=g_ab)

        lossa.create()
        lossb.create()

        cycloss = tf.reduce_mean(tf.abs(self.inputs.xa-cyca)) + \
                       tf.reduce_mean(tf.abs(self.inputs.xb-cycb))

        # loss terms

        cycloss_lambda = config.cycloss_lambda
        if cycloss_lambda is None:
            cycloss_lambda = 10
        cycloss *= cycloss_lambda
        loss1=('generator', cycloss + lossb.g_loss)
        loss2=('discriminator', lossb.d_loss)
        loss3=('generator', cycloss + lossa.g_loss)
        loss4=('discriminator', lossa.d_loss)

        var_lists = []
        var_lists.append(encode_a.variables() + g_ab.variables())
        var_lists.append(discriminator_b.variables())
        var_lists.append(encode_b.variables() + g_ba.variables())
        var_lists.append(discriminator_a.variables())

        metrics = []
        metrics.append(lossa.metrics)
        metrics.append(None)
        metrics.append(lossb.metrics)
        metrics.append(None)

        self.trainer = MultiStepTrainer(self, self.config.trainer, [loss1,loss2,loss3,loss4], var_lists=var_lists, metrics=metrics)
        self.trainer.create()

        self.cyca = cyca
        self.cycb = cycb
        self.cycloss = cycloss
        self.encoder = encode_a
        self.generator = g_ab

        self.session.run(tf.global_variables_initializer())

    def step(self, feed_dict={}):
        return self.trainer.step(feed_dict)

