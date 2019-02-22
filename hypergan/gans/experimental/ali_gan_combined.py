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

class AliGANCombined(BaseGAN):
    """ 
    """
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)

    def required(self):
        """
        `encoder`  encodes X into Z
        `discriminator`  measures X and G.
        `generator` produces samples
        """
        return "generator discriminator ".split()

    def create(self):
        config = self.config
        ops = self.ops
        d_losses = []
        g_losses = []

        def random_like(x):
            return UniformDistribution(self, config.z_distribution, output_shape=self.ops.shape(x)).sample
        with tf.device(self.device):
            x_input = tf.identity(self.inputs.x, name='input')

            # q(z|x)
            if config.u_to_z:
                UniformDistribution = UniformDistribution(self, config.z_distribution)
            else:
                z_shape = self.ops.shape(encoder.sample)
                uz_shape = z_shape
                uz_shape[-1] = uz_shape[-1] // len(config.z_distribution.projections)
                UniformDistribution = UniformDistribution(self, config.z_distribution, output_shape=uz_shape)
            self.uniform_distribution = uniform_encoder
 
            direction, slider = self.create_controls(self.ops.shape(UniformDistribution.sample))
            z = UniformDistribution.sample + slider * direction
            
            #projected_encoder = UniformDistribution(self, config.encoder, z=encoder.sample)


            feature_dim = len(ops.shape(z))-1
            #stack_z = tf.concat([encoder.sample, z], feature_dim)
            #stack_encoded = tf.concat([encoder.sample, encoder.sample], feature_dim)
            stack_z = z

            u_to_z = self.create_component(config.u_to_z, name='u_to_z', input=z)
            generator = self.create_component(config.generator, input=u_to_z.sample, name='generator')
            stacked = [x_input, generator.sample]
            self.generator = generator

            encoder = self.create_encoder(self.inputs.x)

            self.encoder = encoder
            features = [encoder.sample, u_to_z.sample]

            reencode_u_to_z = self.create_encoder(generator.sample, reuse=True)
            print("GEN SAMPLE", generator.sample, reencode_u_to_z.sample, u_to_z.sample)
            reencode_u_to_z_to_g= self.create_component(config.generator, input=reencode_u_to_z.sample, name='generator', reuse=True)
            stacked += [reencode_u_to_z_to_g.sample]
            features += [reencode_u_to_z.sample]

            x_hat = self.create_component(config.generator, input=encoder.sample, reuse=True, name='generator').sample
            self.uniform_sample = generator.sample

            stacked_xg = tf.concat(stacked, axis=0)
            features_zs = tf.concat(features, axis=0)


            standard_discriminator = self.create_component(config.discriminator, name='discriminator', input=stacked_xg, features=[features_zs])
            self.discriminator = standard_discriminator
            d_vars = standard_discriminator.variables()
            g_vars = generator.variables() + encoder.variables()
            g_vars += u_to_z.variables()



            self._g_vars = g_vars
            self._d_vars = d_vars
            standard_loss = self.create_loss(config.loss, standard_discriminator, x_input, generator, len(stacked))
            if self.gan.config.infogan:
                d_vars += self.gan.infogan_q.variables()

            loss1 = ["g_loss", standard_loss.g_loss]
            loss2 = ["d_loss", standard_loss.d_loss]

            d_losses.append(standard_loss.d_loss)
            g_losses.append(standard_loss.g_loss)
            if self.config.autoencode:
                l2_loss = self.ops.squash(10*tf.square(x_hat - x_input))
                g_losses=[l2_loss]
                d_losses=[l2_loss]

            for i,l in enumerate(g_losses):
                self.add_metric('gl'+str(i),l)
            for i,l in enumerate(d_losses):
                self.add_metric('dl'+str(i),l)
            loss = hc.Config({
                'd_fake':standard_loss.d_fake,
                'd_real':standard_loss.d_real,
                'sample': [tf.add_n(d_losses), tf.add_n(g_losses)]
            })
            self.loss = loss
            self.uniform_distribution = uniform_encoder
            trainer = self.create_component(config.trainer, loss = loss, g_vars = g_vars, d_vars = d_vars)

            self.session.run(tf.global_variables_initializer())

        self.trainer = trainer
        self.generator = generator
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



    def fitness_inputs(self):
        return [
                self.uniform_distribution.sample
                ]


    def create_loss(self, loss_config, discriminator, x, generator, split, reuse=False):
        loss = self.create_component(loss_config, discriminator = discriminator, x=x, generator=generator.sample, split=split, reuse=reuse)
        return loss

    def create_controls(self, z_shape):
        direction = tf.random_normal(z_shape, stddev=0.3, name='direction')
        slider = tf.get_variable('slider', initializer=tf.constant_initializer(0.0), shape=[1, 1], dtype=tf.float32, trainable=False)
        return direction, slider

    def create_encoder(self, x_input, name='encoder', reuse=False):
        config = self.config
        encoder = dict(config.encoder or config.g_encoder or config.generator)
        encoder = self.create_component(encoder, name=name, input=x_input, reuse=reuse)
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

    def g_vars(self):
        return self._g_vars
    def d_vars(self):
        return self._d_vars
