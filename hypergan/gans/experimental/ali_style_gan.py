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

class AliStyleGAN(BaseGAN):
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
            x_input = tf.identity(self.inputs.x, name='input')

            def random_like(x):
                return UniformDistribution(self, config.z_distribution, output_shape=self.ops.shape(x)).sample
            # q(z|x)
            encoder = self.create_component(config.encoder, input=x_input, name='z_encoder')
            print("ENCODER ", encoder.sample)

            self.encoder = encoder
            z_shape = self.ops.shape(encoder.sample)
            style = self.create_component(config.style_encoder, input=x_input, name='style')
            self.style = style
            self.styleb = style # hack for sampler
            self.random_style = random_like(style.sample)

            uz_shape = z_shape
            uz_shape[-1] = uz_shape[-1] // len(config.z_distribution.projections)
            UniformDistribution = UniformDistribution(self, config.z_distribution, output_shape=uz_shape)
            direction, slider = self.create_controls(self.ops.shape(UniformDistribution.sample))
            z = UniformDistribution.sample + slider * direction
            
            #projected_encoder = UniformDistribution(self, config.encoder, z=encoder.sample)


            feature_dim = len(ops.shape(z))-1
            #stack_z = tf.concat([encoder.sample, z], feature_dim)
            #stack_encoded = tf.concat([encoder.sample, encoder.sample], feature_dim)
            stack_z = z

            generator = self.create_component(config.generator, input=stack_z, features=[style.sample])
            self.uniform_sample = generator.sample
            x_hat = generator.reuse(encoder.sample)

            features_zs = ops.concat([encoder.sample, z], axis=0)
            stacked_xg = ops.concat([x_input, generator.sample], axis=0)

            if config.u_to_z:
                u_to_z = self.create_component(config.u_to_z, name='u_to_z', input=random_like(z))
                gu = generator.reuse(u_to_z.sample)
                stacked_xg = ops.concat([x_input, gu], axis=0)
                features_zs = ops.concat([encoder.sample, u_to_z.sample], axis=0)


            standard_discriminator = self.create_component(config.discriminator, name='discriminator', input=stacked_xg, features=[features_zs])
            standard_loss = self.create_loss(config.loss, standard_discriminator, x_input, generator, 2)

            l = standard_loss
            d_loss1 = l.d_loss
            g_loss1 = l.g_loss

            d_vars1 = standard_discriminator.variables()
            g_vars1 = generator.variables()+encoder.variables()+style.variables()


            metrics = {
                    'g_loss': l.g_loss,
                    'd_loss': l.d_loss
                }
            t0 = random_like(style.sample)#tf.concat([random_like(style1), random_like(style1)], axis=1)
            t1 = style.sample#tf.concat([style1, style2], axis=1)
            stack = [t0,t1]
            stacked = ops.concat(stack, axis=0)
            features = None
            z_d = self.create_component(config.z_discriminator, name='forcerandom_discriminator', input=stacked)
            loss3 = self.create_component(config.loss, discriminator = z_d, x=x_input, generator=generator, split=2)
            metrics["forcerandom_gloss"]=loss3.g_loss
            metrics["forcerandom_dloss"]=loss3.d_loss
            if config.forcerandom:
                d_loss1 += loss3.d_loss
                g_loss1 += loss3.g_loss
                d_vars1 += z_d.variables()

            if config.u_to_z:
                g_vars1 += u_to_z.variables()

            lossa = hc.Config({'sample': [d_loss1, g_loss1], 'metrics': metrics})
            trainer = ConsensusTrainer(self, config.trainer, loss = lossa, g_vars = g_vars1, d_vars = d_vars1)

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

        d_vars = standard_discriminator.variables()
        g_vars = generator.variables() + encoder.variables()
        print("D_VARS", d_vars)
        print("G_VARS", g_vars)
        #d_loss = standard_loss.d_loss
        #g_loss = standard_loss.g_loss + cycloss
        loss1 = ("g_loss", standard_loss.g_loss)
        loss2 = ("d_loss", standard_loss.d_loss)
        loss = hc.Config({'sample': [standard_loss.d_loss, standard_loss.g_loss], 'metrics': 
            {'g_loss': loss1[1], 'd_loss': loss2[1]}})
        trainer = ConsensusTrainer(self, self.config.trainer, loss = loss, g_vars = g_vars, d_vars = d_vars)
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
