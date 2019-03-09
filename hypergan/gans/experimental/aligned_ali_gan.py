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

class AlignedAliGAN(BaseGAN):
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

            ga = self.create_component(config.generator, input=xb_input, name='a_generator')
            gb = self.create_component(config.generator, input=xa_input, name='b_generator')

            za = ga.controls["z"]
            zb = gb.controls["z"]

            self.uniform_sample = ga.sample

            xba = ga.sample
            xab = gb.sample
            xa_hat = ga.reuse(gb.sample)
            xb_hat = gb.reuse(ga.sample)

            z_shape = self.ops.shape(za)
            uz_shape = z_shape
            uz_shape[-1] = uz_shape[-1] // len(config.z_distribution.projections)
            ue = UniformDistribution(self, config.z_distribution, output_shape=uz_shape)
            features_a = ops.concat([ga.sample, xa_input], axis=0)
            features_b = ops.concat([gb.sample, xb_input], axis=0)
            stacked_a = ops.concat([xa_input, ga.sample], axis=0)
            stacked_b = ops.concat([xb_input, gb.sample], axis=0)
            stacked_z = ops.concat([ue.sample, za, zb], axis=0)
            da = self.create_component(config.discriminator, name='a_discriminator', input=stacked_a, features=[features_b])
            db = self.create_component(config.discriminator, name='b_discriminator', input=stacked_b, features=[features_a])
            dz = self.create_component(config.z_discriminator, name='z_discriminator', input=stacked_z)

            if config.ali_z:
                features_alia = ops.concat([za, ue.sample], axis=0)
                features_alib = ops.concat([zb, ue.sample], axis=0)
                uga = ga.reuse(tf.zeros_like(xb_input), replace_controls={"z":ue.sample})
                ugb = gb.reuse(tf.zeros_like(xa_input), replace_controls={"z":ue.sample})
                stacked_alia = ops.concat([xa_input, uga], axis=0)
                stacked_alib = ops.concat([xb_input, ugb], axis=0)

                dalia = self.create_component(config.ali_discriminator, name='alia_discriminator', input=stacked_alia, features=[features_alib])
                dalib = self.create_component(config.ali_discriminator, name='alib_discriminator', input=stacked_alib, features=[features_alia])
                lalia = self.create_loss(config.loss, dalia, None, None, 2)
                lalib = self.create_loss(config.loss, dalib, None, None, 2)

            la = self.create_loss(config.loss, da, xa_input, ga.sample, 2)
            lb = self.create_loss(config.loss, db, xb_input, gb.sample, 2)
            lz = self.create_loss(config.loss, dz, None, None, 3)

            d_vars = da.variables() + db.variables() + lz.variables()
            if config.ali_z:
                d_vars += dalia.variables() + dalib.variables()
            g_vars = ga.variables() + gb.variables()

            d_loss = la.d_loss+lb.d_loss+lz.d_loss
            g_loss = la.g_loss+lb.g_loss+lz.g_loss
            metrics = {
                    'ga_loss': la.g_loss,
                    'gb_loss': lb.g_loss,
                    'gz_loss': lz.g_loss,
                    'da_loss': la.d_loss,
                    'db_loss': lb.d_loss,
                    'dz_loss': lz.d_loss
                }
            if config.ali_z:
                d_loss+=lalib.d_loss+lalia.d_loss
                g_loss+=lalib.g_loss+lalia.g_loss
                metrics['galia_loss']=lalia.g_loss
                metrics['galib_loss']=lalib.g_loss
                metrics['dalia_loss']=lalia.d_loss
                metrics['dalib_loss']=lalib.d_loss

            loss = hc.Config({'sample': [d_loss, g_loss], 'metrics': metrics})
            trainer = ConsensusTrainer(self, config.trainer, loss = loss, g_vars = g_vars, d_vars = d_vars)
            self.session.run(tf.global_variables_initializer())

        self.trainer = trainer
        self.generator = ga
        self.encoder = gb # this is the other gan
        self.uniform_distribution = hc.Config({"sample":za})#uniform_encoder
        self.zb = zb
        self.z_hat = gb.sample
        self.x_input = xa_input
        self.autoencoded_x = xa_hat

        self.cyca = xa_hat
        self.cycb = xb_hat
        self.xba = xba
        self.xab = xab

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
