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

class AlignedGAN(BaseGAN):
    """ 
    """
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)

    def required(self):
        """
        `input_encoder` is a discriminator.  It encodes X into Z
        `discriminator` is a standard discriminator.  It measures X, reconstruction of X, and G.
        `generator` produces two samples, input_encoder output and a known random distribution.
        """
        return "generator discriminator ".split()

    def create(self):
        config = self.config
        ops = self.ops

        with tf.device(self.device):
            #x_input = tf.identity(self.inputs.x, name='input')
            x_input = tf.identity(self.inputs.xa, name='xa_i')
            z_input = tf.identity(self.inputs.xb, name='xb_i')

            # q(z|x)
            encoder = self.create_encoder(x_input)

            self.encoder = encoder
            z_shape = self.ops.shape(encoder.sample)

            generator = self.create_component(config.generator, input=z_input)
            self.uniform_sample = generator.sample
            if "z" in generator.controls:
                z_control = generator.controls["z"]
            else:
                z_control = generator.sample
            zb = encoder.controls["z"]
            x_hat = generator.reuse(encoder.sample)
            xba = generator.reuse(z_input)
            xab = encoder.reuse(x_input)
            cycb = encoder.reuse(xba)

            features_xg = ops.concat([generator.sample, x_input], axis=0)
            features_zs = ops.concat([encoder.sample, z_input], axis=0)
            stacked_xg = ops.concat([x_input, generator.sample], axis=0)
            stacked_zs = ops.concat([z_input, encoder.sample], axis=0)

            standard_discriminator = self.create_component(config.discriminator, name='discriminator', input=stacked_xg, features=[features_zs])
            z_discriminator = self.create_component(config.discriminator, name='z_discriminator', input=stacked_zs, features=[features_xg])
            #z_discriminator = self.create_z_discriminator(z_input, encoder.sample)
            standard_loss = self.create_loss(config.loss, standard_discriminator, x_input, generator, 2)

            encoder_loss = self.create_loss(config.eloss or config.loss, z_discriminator, z_input, encoder, 2)

            trainer = self.create_trainer(None, None, encoder, generator, encoder_loss, standard_loss, standard_discriminator, z_discriminator)
            self.session.run(tf.global_variables_initializer())

        self.trainer = trainer
        self.generator = generator
        self.uniform_distribution = hc.Config({"sample":z_control})#uniform_encoder
        self.z = z_input
        self.zb = zb
        self.z_hat = encoder.sample
        self.x_input = x_input
        self.autoencoded_x = x_hat

        self.cyca = x_hat
        self.cycb = cycb
        self.xba = xba
        self.xab = xab
        self.uga = self.cyca
        self.ugb = self.cycb

        rgb = tf.cast((self.generator.sample+1)*127.5, tf.int32)
        self.generator_int = tf.bitwise.bitwise_or(rgb, 0xFF000000, name='generator_int')

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
        input_encoder = dict(config.input_encoder or config.g_encoder or config.generator)
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

        metrics = []
        metrics.append(standard_loss.metrics)

        print('ed vars!', encoder_discriminator.variables())
        d_vars = standard_discriminator.variables() + encoder_discriminator.variables()
        g_vars = generator.variables() + encoder.variables()
        print("D_VARS", d_vars)
        print("G_VARS", g_vars)
        #d_loss = standard_loss.d_loss
        #g_loss = standard_loss.g_loss + cycloss
        d_loss = standard_loss.d_loss+encoder_loss.d_loss
        g_loss = standard_loss.g_loss+encoder_loss.g_loss
        loss = hc.Config({'sample': [d_loss, g_loss], 'metrics': 
            {
                'g_loss': standard_loss.g_loss,
                'e_loss': encoder_loss.g_loss,
                'ed_loss': encoder_loss.d_loss,
                'd_loss': standard_loss.d_loss
            }
        })
        trainer = ConsensusTrainer(self, self.config.trainer, loss = loss, g_vars = g_vars, d_vars = d_vars)
        return trainer

    def input_nodes(self):
        "used in hypergan build"
        if hasattr(self.generator, 'mask_generator'):
            extras = [self.mask_generator.sample]
        else:
            extras = []
        return extras + [
                self.x_input
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
                self.generator_int
        ]
