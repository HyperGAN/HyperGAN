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

class AliVibGAN(BaseGAN):
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
    def bottleneck(self, metric, name, term1, term2):
        dvs = []
        _inputs = term1
        inputs = tf.concat(term2, axis=0)
        features = None
        bdisc = self.create_component(config[name+'1'], name=name+'1', input=inputs, features=[features])
        dvs += bdisc.variables()
        l2 = self.create_loss(config.loss, bdisc, None, None, len(_inputs))
        self.add_metric(metric+'_dl1', l2.d_loss)
        self.add_metric(metric+'_gl1', l2.g_loss)
        dl= ib_1_c * l2.d_loss
        gl=ib_1_c * l2.g_loss

        beta = config.bottleneck_beta or 1
        _features = term2
        inputs = tf.concat(_inputs, axis=0)
        features = tf.concat(_features, axis=0)
        bdisc2 = self.create_component(config[name+'2'], name=name+'2', input=inputs, features=[features])
        dvs += bdisc2.variables()
        l2 = self.create_loss(config.loss, bdisc2, None, None, len(_inputs))
        self.add_metric(metric+'_dl2',  ib_2_c * beta * l2.d_loss)
        self.add_metric(metric+'_gl2',  ib_2_c * beta * l2.g_loss)
        dl += ib_2_c * beta * l2.d_loss
        gl += ib_2_c * beta * l2.g_loss
        return gl, dl, dvs

    def create(self):
        config = self.config
        ops = self.ops
        d_losses = []
        g_losses = []
        encoder = self.create_encoder(self.inputs.x)

        with tf.device(self.device):
            x_input = tf.identity(self.inputs.x, name='input')

            if config.u_to_z:
                latent = UniformDistribution(self, config.latent)
            else:
                z_shape = self.ops.shape(encoder.sample)
                uz_shape = z_shape
                uz_shape[-1] = uz_shape[-1] // len(config.latent.projections)
                latent = UniformDistribution(self, config.latent, output_shape=uz_shape)
            self.uniform_distribution = latent 
            self.latent = latent
            direction, slider = self.create_controls(self.ops.shape(latent.sample))
            z = latent.sample + slider * direction

            u_to_z = self.create_component(config.u_to_z, name='u_to_z', input=z)
            generator = self.create_component(config.generator, input=u_to_z.sample, name='generator')
            stacked = [x_input, generator.sample]
            self.generator = generator


            self.encoder = encoder
            features = [encoder.sample, u_to_z.sample]

            reencode_u_to_z = self.create_encoder(generator.sample, reuse=True)
            reencode_u_to_z_to_g= self.create_component(config.generator, input=reencode_u_to_z.sample, name='generator', reuse=True)

            self.reencode_g = reencode_u_to_z_to_g

            x_hat = self.create_component(config.generator, input=encoder.sample, reuse=True, name='generator').sample
            reencode_x_hat_to_z = self.create_encoder(x_hat, reuse=True)
            self.uniform_sample = generator.sample

            d_vars = []
            g_vars = generator.variables() + encoder.variables()
            g_vars += u_to_z.variables()

            def ali(*stack, reuse=False):
                xs=[t for t,_ in stack]
                zs=[t for _,t in stack]
                xs=tf.concat(xs,axis=0)
                zs=tf.concat(zs,axis=0)

                discriminator = self.create_component(config.discriminator, name='discriminator', input=xs, features=[zs], reuse=reuse)
                loss = self.create_loss(config.loss, discriminator, None, None, len(stack))
                return loss,discriminator

            def d(name, stack):
                if name is None:
                    name = config
                stacked = tf.concat(stack,axis=0)
                discriminator = self.create_component(config[name], name=name, input=stacked)
                loss = self.create_loss(config.loss, discriminator, None, None, len(stack))
                return loss,discriminator

            l1, d1 = ali([self.inputs.x,encoder.sample],[generator.sample,u_to_z.sample],[reencode_u_to_z_to_g.sample, reencode_u_to_z.sample])
            l2, d2 = ali([self.inputs.x,tf.zeros_like(encoder.sample)],[generator.sample,tf.zeros_like(u_to_z.sample)],[reencode_u_to_z_to_g.sample, tf.zeros_like(reencode_u_to_z.sample)], reuse=True)
            l3, d3 = ali([tf.zeros_like(self.inputs.x),encoder.sample],[tf.zeros_like(generator.sample),u_to_z.sample],[tf.zeros_like(reencode_u_to_z_to_g.sample), reencode_u_to_z.sample], reuse=True)


            if config.alternate:
                d_losses = [beta * (l1.d_loss - l2.d_loss - l3.d_loss) + l2.d_loss + 2*l3.d_loss]
                g_losses = [beta * (l1.g_loss - l2.g_loss - l3.g_loss) + l2.g_loss + 2*l3.g_loss]

            if config.mutual_only:
                d_losses = [2*l1.d_loss - l2.d_loss - l3.d_loss]
                g_losses = [2*l1.g_loss - l2.g_loss - l3.g_loss]
            else:

                l4, d4 = d('x_discriminator', [self.inputs.x, generator.sample, reencode_u_to_z_to_g.sample])
                l5, d5 = d('z_discriminator', [encoder.sample, u_to_z.sample, reencode_u_to_z.sample])

                beta = config.beta or 0.9
                d_losses = [beta * (l1.d_loss - l2.d_loss - l3.d_loss) + l4.d_loss + 2*l5.d_loss]
                g_losses = [beta * (l1.g_loss - l2.g_loss - l3.g_loss) + l4.g_loss + 2*l5.g_loss]
            self.discriminator = d1
            self.loss = l1


            self.add_metric("ld", d_losses[0])
            self.add_metric("lg", g_losses[0])

            if config.mutual_only or config.alternate:
                for d in [d1]:
                    d_vars += d.variables()
            else:
                for d in [d1,d4,d5]:
                    d_vars += d.variables()

            self._g_vars = g_vars
            self._d_vars = d_vars

            loss = hc.Config({
                'd_fake':l1.d_fake,
                'd_real':l1.d_real,
                'sample': [tf.add_n(d_losses), tf.add_n(g_losses)]
            })
            self.loss = loss
            self.uniform_distribution = latent 
            trainer = self.create_component(config.trainer, g_vars = g_vars, d_vars = d_vars)

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
        self.random_z = tf.random_uniform(ops.shape(latent.sample), -1, 1, name='random_z')

        if hasattr(generator, 'mask_generator'):
            self.mask_generator = generator.mask_generator
            self.mask = mask
            self.autoencode_mask = generator.mask_generator.sample
            self.autoencode_mask_3_channel = generator.mask

    def get_layer(self, name):
        return self.discriminator.named_layers[name]

    def fitness_inputs(self):
        return [
                self.uniform_distribution.sample
                ]


    def create_loss(self, loss_config, discriminator, x, generator, split, reuse=False):
        loss = self.create_component(loss_config, discriminator = discriminator, x=x, split=split, reuse=reuse)
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
