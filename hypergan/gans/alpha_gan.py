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

    def required(self):
        """
        `input_encoder` is a discriminator.  It encodes X into Z
        `z_discriminator` is another discriminator.  It takes as input the output of input_encoder and z
        `discriminator` is a standard discriminator.  It measures X, reconstruction of X, and G.
        `generator` produces two samples, input_encoder output and a known random distribution.
        """
        return "generator discriminator z_discriminator ".split()

    def create(self):
        config = self.config
        ops = self.ops

        with tf.device(self.device):
            x_input = tf.identity(self.inputs.x, name='input')

            encoder = self.create_encoder(x_input)
            z_shape = self.ops.shape(encoder.sample)

            uniform_encoder = UniformEncoder(self, config.encoder, output_shape=z_shape)
            direction, slider = self.create_controls(z_shape)
            z = uniform_encoder.sample + slider * direction

            z_discriminator = self.create_z_discriminator(uniform_encoder.sample, encoder.sample)

            generator = self.create_component(config.generator, input=z)
            if hasattr(generator, 'mask_single_channel'):
                mask = generator.mask_single_channel
            x_hat = generator.reuse(encoder.sample)

            encoder_loss = self.create_loss(config.eloss or config.loss, z_discriminator, z, encoder, 2)

            if config.segments_included:
                g1x_hat = generator.g1x
                g2x_hat = generator.g2x
                generator.reuse(z, mask=generator.mask_single_channel)
                stacked = [x_input, g1x_hat, g2x_hat, generator.g1x, generator.g2x]
            else:
                stacked = [x_input, generator.sample, x_hat]

            stacked_xg = ops.concat(stacked, axis=0)
            standard_discriminator = self.create_component(config.discriminator, name='discriminator', input=stacked_xg)
            standard_loss = self.create_loss(config.loss, standard_discriminator, x_input, generator, len(stacked))

            #loss terms
            cycloss = self.create_cycloss(x_input, x_hat)
            z_cycloss = self.create_z_cycloss(uniform_encoder.sample, encoder.sample, encoder, generator)

            trainer = self.create_trainer(cycloss, z_cycloss, encoder, generator, encoder_loss, standard_loss, standard_discriminator, z_discriminator)
            self.session.run(tf.global_variables_initializer())

        self.trainer = trainer
        self.generator = generator
        self.encoder = encoder
        self.uniform_encoder = uniform_encoder
        self.slider = slider
        self.direction = direction
        self.z = z
        self.z_hat = encoder.sample
        self.x_input = x_input
        self.uniform_sample = generator.sample
        self.autoencoded_x = x_hat

        if hasattr(generator, 'mask_generator'):
            self.mask_generator = generator.mask_generator
            self.mask = mask
            self.autoencode_mask = generator.mask_generator.sample
            self.autoencode_mask_3_channel = generator.mask

    def create_loss(self, loss_config, discriminator, x, generator, split):
        loss = self.create_component(loss_config, discriminator = discriminator, x=x, generator=generator.sample, split=split)
        return loss

    def create_controls(self, z_shape):
        direction = tf.random_normal(z_shape, stddev=0.3, name='direction')
        slider = tf.get_variable('slider', initializer=tf.constant_initializer(0.0), shape=[1, 1], dtype=tf.float32, trainable=False)
        return direction, slider

    def create_encoder(self, x_input):
        config = self.config
        input_encoder = dict(config.input_encoder or config.g_encoder or config.discriminator)
        encoder = self.create_component(input_encoder, name='input_encoder', input=x_input)
        return encoder

    def create_z_discriminator(self, z, z_hat):
        config = self.config
        z_discriminator = dict(config.z_discriminator or config.discriminator)
        z_discriminator['layer_filter']=None
        net = tf.concat(axis=0, values=[z, z_hat])
        encoder_discriminator = self.create_component(z_discriminator, name='z_discriminator', input=net)
        return encoder_discriminator

    def create_cycloss(self, x_input, x_hat):
        config = self.config
        ops = self.ops
        distance = config.distance or ops.lookup('l1_distance')
        cycloss = tf.reduce_mean(distance(x_input,x_hat))
        cycloss_lambda = config.cycloss_lambda
        if cycloss_lambda is None:
            cycloss_lambda = 10

        cycloss *= cycloss_lambda
        return cycloss


    def create_z_cycloss(self, z, x_hat, encoder, generator):
        config = self.config
        ops = self.ops
        total = None
        distance = config.distance or ops.lookup('l1_distance')
        if config.z_hat_lambda:
            z_hat_cycloss_lambda = config.z_hat_cycloss_lambda
            recode_z_hat = encoder.reuse(x_hat)
            z_hat_cycloss = tf.reduce_mean(distance(z_hat,recode_z_hat))
            z_hat_cycloss *= z_hat_cycloss_lambda
        if config.z_cycloss_lambda:
            recode_z = encoder.reuse(generator.reuse(z))
            z_cycloss = tf.reduce_mean(distance(z,recode_z))
            z_cycloss_lambda = config.z_cycloss_lambda
            if z_cycloss_lambda is None:
                z_cycloss_lambda = 0
            z_cycloss *= z_cycloss_lambda

        if config.z_hat_lambda and config.z_cycloss_lambda:
            total = z_cycloss + z_hat_cycloss
        elif config.z_cycloss_lambda:
            total = z_cycloss
        elif config.z_hat_lambda:
            total = z_hat_cycloss
        return total



    def create_trainer(self, cycloss, z_cycloss, encoder, generator, encoder_loss, standard_loss, standard_discriminator, encoder_discriminator):
        if z_cycloss is not None:
            loss1=('generator encoder', z_cycloss + cycloss + encoder_loss.g_loss)
            loss2=('generator image', z_cycloss + cycloss + standard_loss.g_loss)
            loss3=('discriminator image', standard_loss.d_loss)
            loss4=('discriminator encoder', encoder_loss.d_loss)
        else:
            loss1=('generator encoder', cycloss + encoder_loss.g_loss)
            loss2=('generator image',cycloss + standard_loss.g_loss)
            loss3=('discriminator image', standard_loss.d_loss)
            loss4=('discriminator encoder', encoder_loss.d_loss)

        var_lists = []
        var_lists.append(encoder.variables())
        var_lists.append(generator.variables())
        var_lists.append(standard_discriminator.variables())
        var_lists.append(encoder_discriminator.variables())

        metrics = []
        metrics.append(None)
        metrics.append(None)
        metrics.append(standard_loss.metrics)
        metrics.append(encoder_loss.metrics)

        trainer = MultiStepTrainer(self, self.config.trainer, [loss1,loss2,loss3,loss4], var_lists=var_lists, metrics=metrics)
        return trainer

    def input_nodes(self):
        "used in hypergan build"
        return [
                self.x_input,
                self.mask_generator.sample,
                self.slider, 
                self.direction,
                self.uniform_encoder.sample
        ]


    def output_nodes(self):
        "used in hypergan build"
        return [
                self.encoder.sample,
                self.generator.sample, 
                self.uniform_sample,
                self.mask_generator.sample,
                self.generator.g1x,
                self.generator.g2x
        ]
