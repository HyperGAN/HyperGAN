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
from hypergan.trainers.experimental.consensus_trainer import ConsensusTrainer

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
            self.encoder = encoder
            z_shape = self.ops.shape(encoder.sample)

            uz_shape = z_shape
            uz_shape[-1] = uz_shape[-1] // len(config.encoder.projections)
            latent = UniformDistribution(self, config.encoder, output_shape=uz_shape)
            direction, slider = self.create_controls(self.ops.shape(latent.sample))
            z = latent.sample + slider * direction
            
            #projected_encoder = UniformDistribution(self, config.encoder, z=encoder.sample)

            z_discriminator = self.create_z_discriminator(latent.sample, encoder.sample)

            feature_dim = len(ops.shape(z))-1
            #stack_z = tf.concat([encoder.sample, z], feature_dim)
            #stack_encoded = tf.concat([encoder.sample, encoder.sample], feature_dim)
            stack_z = z

            generator = self.create_component(config.generator, input=stack_z)
            self.uniform_sample = generator.sample
            x_hat = generator.reuse(encoder.sample)

            if hasattr(generator, 'mask_single_channel'):
                mask = generator.mask_single_channel

            encoder_loss = self.create_loss(config.eloss or config.loss, z_discriminator, z, encoder, 2)
            if config.segments_included:
                newsample = generator.reuse(stack_z, mask=generator.mask_single_channel)
#                stacked = [x_input, generator.sample, newsample, x_hat]
                stacked = [x_input, newsample, generator.sample, x_hat, generator.g1x, generator.g2x, generator.g3x]
                #stacked = [x_input, g1x, g2x, newsample, generator.sample, x_hat]
                #stacked = [x_input, newsample, generator.sample, x_hat]
            elif config.simple_d:
                stacked = [x_input, self.uniform_sample]
            else:
                stacked = [x_input, self.uniform_sample, x_hat]

            stacked_xg = ops.concat(stacked, axis=0)
            standard_discriminator = self.create_component(config.discriminator, name='discriminator', input=stacked_xg)
            standard_loss = self.create_loss(config.loss, standard_discriminator, x_input, generator, len(stacked))
            self.loss = standard_loss

            #loss terms
            cycloss = self.create_cycloss(x_input, x_hat)
            z_cycloss = self.create_z_cycloss(latent.sample, encoder.sample, encoder, generator)

            #first_pixel = tf.slice(generator.mask_single_channel, [0,0,0,0], [-1,1,1,-1]) + 1 # we want to minimize range -1 to 1
            #cycloss += tf.reduce_sum(tf.reshape(first_pixel, [-1]), axis=0)

            if hasattr(generator, 'mask'): #TODO only segment
                cycloss_whitening_lambda = config.cycloss_whitening_lambda or 0.01
                cycloss += tf.reduce_mean(tf.reshape(0.5-tf.abs(generator.mask-0.5), [-1]), axis=0) * cycloss_whitening_lambda

            #if hasattr(generator, 'mask'): # TODO only multisegment
            #    cycloss_single_channel_lambda = config.cycloss_single_channel_lambda or 0.01
            #    m = tf.reduce_sum(generator.mask, 3)
            #    cycloss += tf.reduce_mean(tf.reshape(tf.abs(1.0-m)/ops.shape(generator.mask)[3], [-1]), axis=0) * cycloss_single_channel_lambda
            if hasattr(generator, 'mask'): # TODO only multisegment
            #    cycloss_single_channel_lambda = config.cycloss_single_channel_lambda or 0.01
                m = tf.reduce_mean(generator.mask, 1, keep_dims=True)
                m = tf.reduce_mean(m, 2, keep_dims=True)
                c = 0.1
                cycloss += (c - tf.minimum(tf.reduce_min(m, 3, keep_dims=True), c))
            #    cycloss += tf.reduce_mean(tf.reshape(tf.abs(1.0-m)/ops.shape(generator.mask)[3], [-1]), axis=0) * cycloss_single_channel_lambda

            trainer = self.create_trainer(cycloss, z_cycloss, encoder, generator, encoder_loss, standard_loss, standard_discriminator, z_discriminator)
            self.session.run(tf.global_variables_initializer())

        self.trainer = trainer
        self.generator = generator
        self.uniform_distribution = uniform_encoder
        self.slider = slider
        self.direction = direction
        self.z = z
        self.z_hat = encoder.sample
        self.x_input = x_input
        self.autoencoded_x = x_hat
        rgb = tf.cast((self.generator.sample+1)*127.5, tf.int32)
        self.generator_int = tf.bitwise.bitwise_or(rgb, 0xFF000000, name='generator_int')
        self.random_z = tf.random_uniform(ops.shape(UniformDistribution.sample), -1, 1, name='random_z')

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

    def create_encoder(self, x_input, name='input_encoder'):
        config = self.config
        input_encoder = dict(config.input_encoder or config.g_encoder or config.discriminator)
        encoder = self.create_component(input_encoder, name=name, input=x_input)
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
        pe_layers = self.gan.skip_connections.get_array("progressive_enhancement")
        cycloss_lambda = config.cycloss_lambda
        if cycloss_lambda is None:
            cycloss_lambda = 10
        
        if(len(pe_layers) > 0):
            mask = self.progressive_growing_mask(len(pe_layers)//2+1)
            cycloss = tf.reduce_mean(distance(mask*x_input,mask*x_hat))

            cycloss *= mask
        else:
            cycloss = tf.reduce_mean(distance(x_input, x_hat))

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

        trainer = self.create_component(self.config.trainer)
        return trainer

    def input_nodes(self):
        "used in hypergan build"
        if hasattr(self.generator, 'mask_generator'):
            extras = [self.mask_generator.sample]
        else:
            extras = []
        return extras + [
                self.x_input,
                self.slider, 
                self.direction,
                self.uniform_distribution.sample
        ]


    def output_nodes(self):
        "used in hypergan build"

    
        if hasattr(self.generator, 'mask_generator'):
            extras = [
                self.mask_generator.sample, 
                self.generator.g1x,
                self.generator.g2x
            ]
        else:
            extras = []
        return extras + [
                self.encoder.sample,
                self.generator.sample, 
                self.uniform_sample,
                self.generator_int,
                self.random_z
        ]
