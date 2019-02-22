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

class AlignedAliOneGAN(BaseGAN):
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

            if config.same_g:
                ga = self.create_component(config.generator, input=xb_input, name='a_generator')
                gb = hc.Config({"sample":ga.reuse(xa_input),"controls":{"z":ga.controls['z']}, "reuse": ga.reuse})
            else:
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
            ue2 = UniformDistribution(self, config.z_distribution, output_shape=uz_shape)
            ue3 = UniformDistribution(self, config.z_distribution, output_shape=uz_shape)
            ue4 = UniformDistribution(self, config.z_distribution, output_shape=uz_shape)
            print('ue', ue.sample)

            zua = ue.sample
            zub = ue2.sample

            uga = ga.reuse(tf.zeros_like(xb_input), replace_controls={"z":zua})
            ugb = gb.reuse(tf.zeros_like(xa_input), replace_controls={"z":zub})

            xa = xa_input
            xb = xb_input

            t0 = ops.concat([xb, xa], axis=3)
            t1 = ops.concat([ugb, uga], axis=3)
            t2 = ops.concat([gb.sample, ga.sample], axis=3)
            f0 = ops.concat([za, zb], axis=3)
            f1 = ops.concat([zub, zua], axis=3)
            f2 = ops.concat([zb, za], axis=3)
            features = ops.concat([f0, f1, f2], axis=0)
            stack = [t0, t1, t2]


            if config.mess2:
                xbxa = ops.concat([xb_input, xa_input], axis=3)
                gbga = ops.concat([gb.sample, ga.sample], axis=3)
                fa = ops.concat([za, zb], axis=3)
                fb = ops.concat([za, zb], axis=3)
                features = ops.concat([fa, fb], axis=0)
                stack = [xbxa, gbga]
 
            if config.mess6:
                t0 = ops.concat([xb, xa], axis=3)

                t1 = ops.concat([gb.sample, uga], axis=3)
                t2 = ops.concat([gb.sample, xa], axis=3)
                t3 = ops.concat([xb, ga.sample], axis=3)
                t4 = ops.concat([ugb, ga.sample], axis=3)
                features = None
                stack = [t0, t1, t2, t3, t4]

            if config.mess7:
                ugb = gb.reuse(tf.zeros_like(xa_input), replace_controls={"z":zua})
                t0 = ops.concat([xb, ga.sample], axis=3)
                t1 = ops.concat([ugb, uga], axis=3)
                t2 = ops.concat([gb.sample, xa], axis=3)
                features = None
                stack = [t0, t1, t2]
 
            if config.mess8:
                ugb = gb.reuse(tf.zeros_like(xa_input), replace_controls={"z":zua})
                uga2 = ga.reuse(tf.zeros_like(xa_input), replace_controls={"z":zub})
                ugb2 = gb.reuse(tf.zeros_like(xa_input), replace_controls={"z":zub})
                t0 = ops.concat([xb, ga.sample, xa, gb.sample], axis=3)
                t1 = ops.concat([ugb, uga, uga2, ugb2], axis=3)
                features = None
                stack = [t0, t1]


            if config.mess10:
                t0 = ops.concat([xb, xa], axis=3)

                ugb = gb.reuse(tf.zeros_like(xa_input), replace_controls={"z":zua})
                t1 = ops.concat([ugb, uga], axis=3)
                t2 = ops.concat([gb.sample, xa], axis=3)
                t3 = ops.concat([xb, ga.sample], axis=3)
                features = None
                stack = [t0, t1, t2, t3]

            if config.mess11:
                t0 = ops.concat([xa, xb, ga.sample, gb.sample], axis=3)
                ugbga = ga.reuse(ugb)
                ugagb = gb.reuse(uga)

                t1 = ops.concat([ga.sample, gb.sample, uga, ugb], axis=3)
                features = None
                stack = [t0, t1]

            if config.mess12:
                t0 = ops.concat([xb, xa], axis=3)
                t2 = ops.concat([gb.sample, ga.sample], axis=3)
                f0 = ops.concat([za, zb], axis=3)
                f2 = ops.concat([zua, zub], axis=3)
                features = ops.concat([f0, f2], axis=0)
                stack = [t0, t2]

            if config.mess13:
                ugb = gb.reuse(tf.zeros_like(xa_input), replace_controls={"z":zua})
                features = None
                t0 = ops.concat([xa, gb.sample], axis=3)
                t1 = ops.concat([ga.sample, xb], axis=3)
                t2 = ops.concat([uga, ugb], axis=3)
                stack = [t0, t1, t2]


            if config.mess14:
                features = None
                t0 = ops.concat([xa, gb.sample], axis=3)
                t1 = ops.concat([ga.sample, xb], axis=3)
                stack = [t0, t1]




            stacked = ops.concat(stack, axis=0)
            d = self.create_component(config.discriminator, name='alia_discriminator', input=stacked, features=[features])
            l = self.create_loss(config.loss, d, xa_input, ga.sample, len(stack))

            d_vars = d.variables()
            if config.same_g:
                g_vars = ga.variables()
            else:
                g_vars = ga.variables() + gb.variables()

            d_loss = l.d_loss
            g_loss = l.g_loss

            metrics = {
                    'g_loss': l.g_loss,
                    'd_loss': l.d_loss
                }

            if(config.alpha):
                #t0 = ops.concat([zua,zub], axis=3)
                #t1 = ops.concat([za,zb], axis=3)
                t0 = zua
                t1 = za
                t2 = zb
                netzd = tf.concat(axis=0, values=[t0,t1,t2])
                z_d = self.create_component(config.z_discriminator, name='z_discriminator', input=netzd)

                print("Z_D", z_d)
                lz = self.create_component(config.loss, discriminator = z_d, x=xa_input, generator=ga, split=2)
                d_loss += lz.d_loss
                g_loss += lz.g_loss
                d_vars += z_d.variables()
                metrics["a_gloss"]=lz.g_loss
                metrics["a_dloss"]=lz.d_loss

            if(config.mess13):
                t0 = ops.concat([xb, ga.sample], axis=3)
                t1 = ops.concat([gb.sample, xa], axis=3)
                t2 = ops.concat([ugb, uga], axis=3)
                stack = [t0, t1, t2]
                features = None
                stacked = tf.concat(axis=0, values=stack)
                d2 = self.create_component(config.discriminator, name='align_2', input=stacked, features=[features])
                lz = self.create_loss(config.loss, d2, xa_input, ga.sample, len(stack))
                d_vars += d2.variables()

                d_loss += lz.d_loss
                g_loss += lz.g_loss
                metrics["mess13_g"]=lz.g_loss
                metrics["mess13_d"]=lz.d_loss

            if(config.mess14):
                t0 = ops.concat([xb, xa], axis=3)
                t2 = ops.concat([ugb, uga], axis=3)
                stack = [t0, t2]
                features = None
                stacked = tf.concat(axis=0, values=stack)
                d3 = self.create_component(config.discriminator, name='align_3', input=stacked, features=[features])
                lz = self.create_loss(config.loss, d3, xa_input, ga.sample, len(stack))
                d_vars += d3.variables()

                d_loss += lz.d_loss
                g_loss += lz.g_loss
                metrics["mess14_g"]=lz.g_loss
                metrics["mess14_d"]=lz.d_loss



            loss = hc.Config({'sample': [d_loss, g_loss], 'metrics': metrics})
            trainer = ConsensusTrainer(self, config.trainer, loss = loss, g_vars = g_vars, d_vars = d_vars)
            self.session.run(tf.global_variables_initializer())

        self.trainer = trainer
        self.generator = ga
        self.encoder = gb # this is the other gan
        self.uniform_distribution = hc.Config({"sample":zb})#uniform_encoder
        self.zb = zb
        self.z_hat = gb.sample
        self.x_input = xa_input
        self.autoencoded_x = xa_hat

        self.cyca = xa_hat
        self.cycb = xb_hat
        self.xba = xba
        self.xab = xab
        self.uga = uga
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
