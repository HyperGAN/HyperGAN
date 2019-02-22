
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

class ConditionalGAN(BaseGAN):
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
            xa_input = tf.identity(self.inputs.xa, name='xa_i')
            xb_input = tf.identity(self.inputs.xb, name='xb_i')

            def random_like(x):
                return UniformDistribution(self, config.z_distribution, output_shape=self.ops.shape(x)).sample
            #y=a
            #x=b
            zgx = self.create_component(config.encoder, input=xa_input, name='xa_to_x')
            zgy = self.create_component(config.encoder, input=xb_input, name='xb_to_y')
            zx = zgx.sample
            zy = zgy.sample

            z_noise = random_like(zx)
            n_noise = random_like(zx)
            if config.style:
                stylex = self.create_component(config.style_discriminator, input=xb_input, name='xb_style')
                styley = self.create_component(config.style_discriminator, input=xa_input, name='xa_style')
                zy = tf.concat(values=[zy, z_noise], axis=3)
                zx = tf.concat(values=[zx, n_noise], axis=3)
                gy = self.create_component(config.generator, features=[styley.sample], input=zy, name='gy_generator')
                y = hc.Config({"sample": xa_input})
                zx = self.create_component(config.encoder, input=y.sample, name='xa_to_x', reuse=True).sample
                zx = tf.concat(values=[zx, z_noise], axis=3)
                gx = self.create_component(config.generator, features=[stylex.sample], input=zx, name='gx_generator')
            else:
                gy = self.create_component(config.generator, features=[z_noise], input=zy, name='gy_generator')
                y = hc.Config({"sample": xa_input})
                zx = self.create_component(config.encoder, input=y.sample, name='xa_to_x', reuse=True).sample
                gx = self.create_component(config.generator, features=[z_noise], input=zx, name='gx_generator')
                stylex=hc.Config({"sample":random_like(y.sample)})

            self.y = y
            self.gy = gy
            self.gx = gx

            ga = gy
            gb = gx

            self.uniform_sample = gb.sample

            xba = ga.sample
            xab = gb.sample
            xa_hat = ga.reuse(zx)
            xb_hat = gb.reuse(zy)
            xa = xa_input
            xb = xb_input

            self.styleb = stylex
            self.random_style = random_like(stylex.sample)



            t0 = xb
            t1 = gx.sample
            f0 = gy.sample
            f1 = y.sample
            stack = [t0, t1]
            stacked = ops.concat(stack, axis=0)
            features = ops.concat([f0, f1], axis=0)
            self.inputs.x = xa
            ugb = gb.reuse(random_like(zy))
            zub = zy
            sourcezub = zy


            d = self.create_component(config.discriminator, name='d_ab', 
                    input=stacked, features=[features])
            l = self.create_loss(config.loss, d, xa_input, ga.sample, len(stack))
            loss1 = l
            d_loss1 = l.d_loss
            g_loss1 = l.g_loss

            d_vars1 = d.variables()
            g_vars1 = gb.variables()+ga.variables()+zgx.variables()+zgy.variables()

            d_loss = l.d_loss
            g_loss = l.g_loss


            metrics = {
                    'g_loss': l.g_loss,
                    'd_loss': l.d_loss
                }


            trainers = []

            lossa = hc.Config({'sample': [d_loss1, g_loss1], 'metrics': metrics})
            #lossb = hc.Config({'sample': [d_loss2, g_loss2], 'metrics': metrics})
            trainers += [ConsensusTrainer(self, config.trainer, loss = lossa, g_vars = g_vars1, d_vars = d_vars1)]
            #trainers += [ConsensusTrainer(self, config.trainer, loss = lossb, g_vars = g_vars2, d_vars = d_vars2)]
            trainer = MultiTrainerTrainer(trainers)
            self.session.run(tf.global_variables_initializer())

        self.trainer = trainer
        self.generator = gb
        self.encoder = hc.Config({"sample":ugb}) # this is the other gan
        self.uniform_distribution = hc.Config({"sample":zub})#uniform_encoder
        self.uniform_distribution_source = hc.Config({"sample":sourcezub})#uniform_encoder
        self.zb = zy
        self.z_hat = gb.sample
        self.x_input = xa_input
        self.autoencoded_x = xb_hat

        self.cyca = xa_hat
        self.cycb = xb_hat
        self.xba = xba
        self.xab = xab
        self.uga = y.sample
        self.ugb = ugb

        rgb = tf.cast((self.generator.sample+1)*127.5, tf.int32)
        self.generator_int = tf.bitwise.bitwise_or(rgb, 0xFF000000, name='generator_int')


    def create_loss(self, loss_config, discriminator, x, generator, split):
        loss = self.create_component(loss_config, discriminator = discriminator, x=x, generator=generator, split=split)
        return loss

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
