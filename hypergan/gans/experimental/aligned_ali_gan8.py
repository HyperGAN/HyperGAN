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

class AlignedAliGAN8(BaseGAN):
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
            def random_like(x):
                return UniformDistribution(self, config.latent, output_shape=self.ops.shape(x)).sample
            self.latent = self.create_component(config.latent, name='forcerandom_discriminator')
            zga = self.create_component(config.encoder, input=self.inputs.xb, name='xb_to_za')
            zgb = self.create_component(config.encoder, input=self.inputs.xa, name='xa_to_zb')
            self.zga = zga
            self.zgb = zgb
            za = zga.sample
            zb = zgb.sample
            if config.style:
                styleb = self.create_component(config.style_encoder, input=self.inputs.xb, name='xb_style')
                stylea = self.create_component(config.style_encoder, input=self.inputs.xa, name='xa_style')

                self.stylea = stylea
                self.styleb = styleb
                self.random_style = random_like(styleb.sample)
                ga = self.create_component(config.generator, input=za, name='a_generator', features=[stylea.sample])
                gb = self.create_component(config.generator, input=zb, name='b_generator', features=[styleb.sample])
            elif config.skip_connections:
                ga = self.create_component(config.generator, input=za, skip_connections=zga.layers, name='a_generator')
                gb = self.create_component(config.generator, input=zb, skip_connections=zgb.layers, name='b_generator')
            else:
                ga = self.create_component(config.generator, input=za, name='a_generator')
                gb = self.create_component(config.generator, input=zb, name='b_generator')
                self.ga = ga
                self.gb = gb


            re_zb = self.create_component(config.encoder, input=ga.sample, name='xa_to_zb', reuse=True)
            re_za = self.create_component(config.encoder, input=gb.sample, name='xb_to_za', reuse=True)
            self.ga = ga
            self.gb = gb

            self.uniform_sample = gb.sample

            xba = ga.sample
            xab = gb.sample
            xa_hat = ga.reuse(re_za.sample)
            xb_hat = gb.reuse(re_zb.sample)
            xa = self.inputs.xa
            xb = self.inputs.xb


            t0 = xb
            t1 = gb.sample
            f0 = re_zb.sample#za
            f1 = zb
            stack = [t0, t1]
            stacked = ops.concat(stack, axis=0)
            features = ops.concat([f0, f1], axis=0)
            self.features = features
            # self.inputs.x = xa
            ugb = gb.sample#gb.reuse(random_like(zb))
            zub = zb
            sourcezub = zb

            #skip_connections = []
            #for (a,b) in zip(zga.layers,zgb.layers):
            #    layer = tf.concat([a,b],axis=0)
            #    skip_connections += [layer]

            d = self.create_component(config.discriminator, name='d_ab', 
                    #skip_connections=skip_connections,
                    input=stacked, features=[features])
            self.discriminator = d
            l = self.create_loss(config.loss, d, self.inputs.xa, ga.sample, len(stack))
            self.loss = l
            loss1 = l
            d_loss1 = l.d_loss
            g_loss1 = l.g_loss

            d_vars1 = d.variables()
            g_vars1 = gb.variables()+zga.variables()+zgb.variables()
            self.generator = gb

            d_loss = l.d_loss
            g_loss = l.g_loss


            metrics = {
                    'g_loss': l.g_loss,
                    'd_loss': l.d_loss
                }

            if config.uga:
                t0 = xa
                t1 = ga.sample
                stack = [t0, t1]
                stacked = ops.concat(stack, axis=0)
                features = None
                z_d = self.create_component(config.discriminator, name='uga_discriminator', input=stacked)
                loss3 = self.create_component(config.loss, discriminator = z_d, x=self.inputs.xa, generator=ga, split=2)
                metrics["uga_gloss"]=loss3.g_loss
                metrics["uga_dloss"]=loss3.d_loss
                d_loss1 += loss3.d_loss
                g_loss1 += loss3.g_loss
                d_vars1 += z_d.variables()


            if config.ugb:
                t0 = xb
                t1 = gb.sample
                stack = [t0, t1]
                stacked = ops.concat(stack, axis=0)
                features = None
                z_d = self.create_component(config.discriminator, name='ugb_discriminator', input=stacked)
                loss3 = self.create_component(config.loss, discriminator = z_d, x=self.inputs.xa, generator=ga, split=2)
                self.db = z_d
                self.lb = loss3
                metrics["ugb_gloss"]=loss3.g_loss
                metrics["ugb_dloss"]=loss3.d_loss
                d_loss1 += loss3.d_loss
                g_loss1 += loss3.g_loss
                d_vars1 += z_d.variables()


            if config.uga2:
                t0 = xa
                t1 = ga.sample
                t2 = ga.reuse(za)
                stack = [t0, t1, t2]
                stacked = ops.concat(stack, axis=0)
                features = None
                z_d = self.create_component(config.discriminator, name='uga_discriminator', input=stacked)
                loss3 = self.create_component(config.loss, discriminator = z_d, x=self.inputs.xa, generator=ga, split=3)
                metrics["uga_gloss"]=loss3.g_loss
                metrics["uga_dloss"]=loss3.d_loss
                d_loss1 += loss3.d_loss
                g_loss1 += loss3.g_loss
                d_vars1 += z_d.variables()


            if config.ugb2:
                t0 = xb
                t1 = gb.sample
                t2 = gb.reuse(zb)
                stack = [t0, t1]
                stacked = ops.concat(stack, axis=0)
                features = None
                z_d = self.create_component(config.discriminator, name='ugb_discriminator', input=stacked)
                loss3 = self.create_component(config.loss, discriminator = z_d, x=self.inputs.xa, generator=ga, split=3)
                metrics["ugb_gloss"]=loss3.g_loss
                metrics["ugb_dloss"]=loss3.d_loss
                d_loss1 += loss3.d_loss
                g_loss1 += loss3.g_loss
                d_vars1 += z_d.variables()

            if config.forcerandom:
                t0 = random_like(styleb.sample)#tf.concat([random_like(style1), random_like(style1)], axis=1)
                t1 = styleb.sample#tf.concat([style1, style2], axis=1)
                stack = [t0,t1]
                stacked = ops.concat(stack, axis=0)
                features = None
                z_d = self.create_component(config.latent, name='forcerandom_discriminator', input=stacked)
                loss3 = self.create_component(config.loss, discriminator = z_d, x=self.inputs.xa, generator=ga, split=2)
                metrics["forcerandom_gloss"]=loss3.g_loss
                metrics["forcerandom_dloss"]=loss3.d_loss
                d_loss1 += loss3.d_loss
                g_loss1 += loss3.g_loss
                d_vars1 += z_d.variables()


            if config.forcerandom2:
                style_reader = self.create_component(config.latent, name='style_discriminator2', input=zb)
                style1 = style_reader.sample
                t0 = random_like(style1)
                t1 = style1
                stack = [t0,t1]
                stacked = ops.concat(stack, axis=0)
                features = None
                z_d = self.create_component(config.latent, name='forcerandom_discriminator2', input=stacked)
                loss3 = self.create_component(config.loss, discriminator = z_d, x=self.inputs.xa, generator=ga, split=2)
                metrics["forcerandom2_gloss"]=loss3.g_loss
                metrics["forcerandom2_dloss"]=loss3.d_loss
                d_loss1 += loss3.d_loss
                g_loss1 += loss3.g_loss
                d_vars1 += z_d.variables()
                g_vars1 += style_reader.variables()


 
            trainers = []
            if config.alpha:
                t0 = random_like(zub)
                t1 = zb
                t2 = za
                netzd = tf.concat(axis=0, values=[t0,t1,t2])
                z_d = self.create_component(config.latent, name='latent', input=netzd)
                loss3 = self.create_component(config.loss, discriminator = z_d, x=self.inputs.xa, generator=ga, split=3)
                metrics["za_gloss"]=loss3.g_loss
                metrics["za_dloss"]=loss3.d_loss
                d_loss1 += loss3.d_loss
                g_loss1 += loss3.g_loss
                d_vars1 += z_d.variables()

            if config.mirror_joint:
                t0 = xa
                t1 = ga.sample
                f0 = zb
                f1 = za
                stack = [t0, t1]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1], axis=0)
                uga = ga.sample
                zua = za
                z_d = self.create_component(config.discriminator, name='d_ba', input=stacked, features=[features])
                loss3 = self.create_component(config.loss, discriminator = z_d, split=2)
                self.gan.add_metric("ba_gloss",loss3.g_loss)
                self.gan.add_metric("ba_dloss",loss3.d_loss)
                d_loss1 += loss3.d_loss
                g_loss1 += loss3.g_loss
                d_vars1 += z_d.variables()
                g_vars1 += ga.variables()
                self.al = loss1
                self.bl = loss3
                self.bd = z_d
                self.ad = d

            if config.style:
                g_vars1 += styleb.variables()

            self._g_vars = g_vars1
            self._d_vars = d_vars1

            self.loss = hc.Config({
                'd_fake':l.d_fake,
                'd_real':l.d_real,
                'sample': [d_loss1, g_loss1],
                'metrics': metrics
                })
            print("g_vars1", g_vars1)
            trainer = self.create_component(config.trainer)

            self.session.run(tf.global_variables_initializer())

        self.trainer = trainer
        self.generator = gb
        self.encoder = hc.Config({"sample":ugb}) # this is the other gan
        self.uniform_distribution = hc.Config({"sample":zub})#uniform_encoder
        self.uniform_distribution_source = hc.Config({"sample":sourcezub})#uniform_encoder
        self.zb = zb
        self.z_hat = gb.sample
        self.x_input = self.inputs.xa
        self.autoencoded_x = xb_hat

        self.cyca = xa_hat
        self.cycb = xb_hat
        self.xba = xba
        self.xab = xab
        self.uga = ugb
        self.ugb = ugb

        rgb = tf.cast((self.generator.sample+1)*127.5, tf.int32)
        self.generator_int = tf.bitwise.bitwise_or(rgb, 0xFF000000, name='generator_int')

    def d_vars(self):
        return self._d_vars
    def g_vars(self):
        return self._g_vars
    def fitness_inputs(self):
        return [
                self.uniform_distribution.sample, self.inputs.xa
                ]



    def create_loss(self, loss_config, discriminator, x, generator, split):
        loss = self.create_component(loss_config, discriminator = discriminator, x=x, generator=generator, split=split)
        return loss

    def create_encoder(self, x_input, name='input_encoder'):
        config = self.config
        input_encoder = dict(config.input_encoder or config.g_encoder or config.generator)
        encoder = self.create_component(input_encoder, name=name, input=x_input)
        return encoder

    def create_latent(self, z, z_hat):
        config = self.config
        latent = dict(config.latent or config.discriminator)
        latent['layer_filter']=None
        net = tf.concat(axis=0, values=[z, z_hat])
        encoder_discriminator = self.create_component(latent, name='latent', input=net)
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
