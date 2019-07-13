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
from .base_gan import BaseGAN

from hypergan.distributions.uniform_distribution import UniformDistribution
from hypergan.trainers.experimental.consensus_trainer import ConsensusTrainer

class AliGAN(BaseGAN):
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
            return UniformDistribution(self, config.latent, output_shape=self.ops.shape(x)).sample
        with tf.device(self.device):
            x_input = tf.identity(self.inputs.x, name='input')

            # q(z|x)
            latent = UniformDistribution(self, config.latent)
            self.latent = latent
            encoder = self.create_encoder(self.inputs.x)
            self.encoder = encoder
 
            direction, slider = self.create_controls(self.ops.shape(latent.sample))
            z = latent.sample + slider * direction
            #projected_encoder = UniformDistribution(self, config.encoder, z=encoder.sample)


            feature_dim = len(ops.shape(z))-1
            #stack_z = tf.concat([encoder.sample, z], feature_dim)
            #stack_encoded = tf.concat([encoder.sample, encoder.sample], feature_dim)


            if config.u_to_z:
                if config.style_encoder:
                    style_encoder = self.create_component(config.style_encoder, input=x_input, name='style_encoder')
                    style = style_encoder.sample
                    #style_sample = tf.concat(style, axis=0)
                    style_sample = style
                    #style_sample=random_like(style_sample)
                    #x_hat_style = style_sample
                    x_hat_style = random_like(style_sample)
                    #style_sample =  random_like(x_hat_style)
                    u_to_z = self.create_component(config.u_to_z, name='u_to_z', features=[style_sample], input=z)
                    generator = self.create_component(config.generator, input=u_to_z.sample, features=[style_sample], name='generator')
                else:
                    u_to_z = self.create_component(config.u_to_z, name='u_to_z', input=z)
                    generator = self.create_component(config.generator, input=u_to_z.sample, name='generator')
                stacked = [x_input, generator.sample]
                self.generator = generator

                self.encoder = encoder
                features = [encoder.sample, u_to_z.sample]
                if self.config.reencode:
                    er = self.create_encoder(generator.sample, reuse=True)
                    features = [encoder.sample, er.sample]

                self.u_to_z = u_to_z
            else:
                er = encoder.sample
                generator = self.create_component(config.generator, input=er, name='generator')
                if self.config.reencode:
                    self.reencode = self.create_encoder(generator.sample, reuse=True)
                    er = self.reencode.sample
                self.generator = generator
                stacked = [x_input, generator.sample]

                self.encoder = encoder
                features = ops.concat([encoder.sample, er], axis=0)

            if config.style_encoder:
                x_hat = self.create_component(config.generator, input=encoder.sample, features=[x_hat_style], reuse=True, name='generator').sample
                stacked += [x_hat]
                features += [encoder.sample]
            else:
                x_hat = self.create_component(config.generator, input=encoder.sample, reuse=True, name='generator').sample
            self.autoencoded_x = x_hat
            self.uniform_sample = generator.sample

            stacked_xg = tf.concat(stacked, axis=0)
            features_zs = tf.concat(features, axis=0)
            self.features = features_zs

            standard_discriminator = self.create_component(config.discriminator, name='discriminator', input=stacked_xg, features=[features_zs])
            self.discriminator = standard_discriminator

            d_vars = standard_discriminator.variables()
            g_vars = generator.variables() + encoder.variables()
            if config.style_encoder:
                g_vars += style_encoder.variables()
            if config.u_to_z:
                g_vars += u_to_z.variables()

            if self.config.manifold_guided:
                reencode_u_to_z = self.create_encoder(generator.sample, reuse=True)
                #stack_z = [encoder.sample, u_to_z.sample]#reencode_u_to_z.sample]
                if self.config.manifold_target == 'reencode_u_to_z':
                    stack_z = [encoder.sample, reencode_u_to_z.sample]
                else:
                    stack_z = [encoder.sample, u_to_z.sample]#reencode_u_to_z.sample]
                if self.config.terms == "eg:ex":
                    stack_z = [reencode_u_to_z.sample, encoder.sample]
                if self.config.terms == "ex:eg":
                    stack_z = [encoder.sample, reencode_u_to_z.sample]
                if self.config.stop_gradients:
                    stack_z = [tf.stop_gradient(encoder.sample), u_to_z.sample]#reencode_u_to_z.sample]
                stacked_zs = ops.concat(stack_z, axis=0)
                z_discriminator = self.create_component(config.z_discriminator, name='z_discriminator', input=stacked_zs)
                self.z_discriminator = z_discriminator
                d_vars += z_discriminator.variables()

            self._g_vars = g_vars
            self._d_vars = d_vars
            standard_loss = self.create_loss(config.loss, standard_discriminator, x_input, generator, len(stacked))
            self.standard_loss = standard_loss

            loss1 = ["g_loss", standard_loss.g_loss]
            loss2 = ["d_loss", standard_loss.d_loss]

            if self.config.manifold_guided:
                l2 = self.create_loss(config.loss, z_discriminator, x_input, generator, len(stack_z), reuse=True)
                d_losses.append(l2.d_loss)
                g_losses.append(l2.g_loss)

            d_losses.append(standard_loss.d_loss)
            g_losses.append(standard_loss.g_loss)
            if self.config.autoencode:
                l2_loss = self.ops.squash(10*tf.square(x_hat - x_input))
                g_losses=[l2_loss]
                d_losses=[l2_loss]
            if self.config.vae:
                mu,sigma = self.encoder.variational
                eps = 1e-8
                lam = config.vae_lambda or 0.001
                latent_loss = lam*(0.5 *self.ops.squash(tf.square(mu)-tf.square(sigma) - tf.log(tf.square(sigma)+eps) - 1, tf.reduce_sum ))
                g_losses.append(latent_loss)
                mu,sigma = u_to_z.variational
                latent_loss = lam*(0.5 *self.ops.squash(tf.square(mu)-tf.square(sigma) - tf.log(tf.square(sigma)+eps) - 1, tf.reduce_sum ))
                g_losses.append(latent_loss)


            for i,l in enumerate(g_losses):
                self.add_metric('gl'+str(i), l)
            for i,l in enumerate(d_losses):
                self.add_metric('dl'+str(i),l)
            loss = hc.Config({
                'd_fake':standard_loss.d_fake,
                'd_real':standard_loss.d_real,
                'sample': [tf.add_n(d_losses), tf.add_n(g_losses)]
            })
            self.loss = loss
            trainer = self.create_component(config.trainer, g_vars = g_vars, d_vars = d_vars)

            self.session.run(tf.global_variables_initializer())

        self.trainer = trainer
        self.generator = generator
        self.slider = slider
        self.direction = direction
        self.z = z
        self.z_hat = encoder.sample
        self.x_input = x_input
        rgb = tf.cast((self.generator.sample+1)*127.5, tf.int32)
        self.generator_int = tf.bitwise.bitwise_or(rgb, 0xFF000000, name='generator_int')
        self.random_z = tf.random_uniform(ops.shape(latent.sample), -1, 1, name='random_z')

        if hasattr(generator, 'mask_generator'):
            self.mask_generator = generator.mask_generator
            self.mask = mask
            self.autoencode_mask = generator.mask_generator.sample
            self.autoencode_mask_3_channel = generator.mask


    def create_discriminator(self, _input, reuse=False):
        return self.create_component(self.config.discriminator, name='discriminator', input=_input, features=[tf.zeros_like(self.features)], reuse=reuse)
    def fitness_inputs(self):
        return [
                self.latent.sample
                ]

    def l1_distance(self):
        return self.inputs.x - self.autoencoded_x


    def create_generator(self, _input, reuse=False):
        config = self.config
        if self.ops.shape(_input) == self.ops.shape(self.gan.latent.sample):
            u_to_z = self.create_component(config.u_to_z, name='u_to_z', input=_input, reuse=reuse)
            generator = self.create_component(config.generator, input=u_to_z.sample, name='generator', reuse=reuse)
        else:
            generator = self.create_component(config.generator, input=_input, name='generator', reuse=reuse)

        return generator

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
                self.latent.sample
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
