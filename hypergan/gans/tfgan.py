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
import tensorflow_gan as tfgan
import hypergan as hg

from hypergan.gan_component import ValidationException, GANComponent
from .base_gan import BaseGAN

class TFGAN(BaseGAN):
    """
    TFGAN makes HG compatible with tensorflow-gan
    """
    def __init__(self, *args, **kwargs):
        self.discriminator = None
        self.latent = None
        self.generator = None
        self.loss = None
        self.trainer = None
        self.features = []
        self.create_trainer = True
        if "create_trainer" in kwargs:
            self.create_trainer = kwargs["create_trainer"]
            del kwargs["create_trainer"]
        BaseGAN.__init__(self, *args, **kwargs)

    def required(self):
        return "generator".split()

    def create(self):
        pass

    def tpu_gan_estimator(self, config):
        def create_latent(is_training=True):
            latent = self.create_component(self.config.z_distribution or config.latent)
            return latent.sample

        def create_generator(z, is_training=True):
            gen = self.create_component(self.config.generator, name="generator", input=z).sample
            self.generator = gen
            return gen

        def create_discriminator(x, is_training=True):
            print("INPUT", x)
            disc = self.create_component(self.config.discriminator, name="discriminator", input=x)
            print("S<", disc.variables())
            self.discriminator = disc
            self.loss = self.create_component(self.config.loss, discriminator=self.discriminator)
            self.losses = [self.loss]

            return disc.sample
        optimizer = self.create_optimizer(self.config.optimizer)
        return tfgan.estimator.TPUGANEstimator(
            generator_fn=create_generator,
            discriminator_fn=create_discriminator,
            generator_loss_fn=tfgan.losses.modified_generator_loss,
            discriminator_loss_fn=tfgan.losses.modified_discriminator_loss,
            generator_optimizer=optimizer,#tf.train.AdamOptimizer(2e-4, 0.5),
            discriminator_optimizer=optimizer,#tf.train.AdamOptimizer(2e-4, 0.5),
            joint_train=True,  # train G and D jointly instead of sequentially.
            train_batch_size=self.args.batch_size,
            predict_batch_size=16,
            use_tpu=True,
            config=config)
       

    def g_vars(self):
        return []
    def d_vars(self):
        return []
