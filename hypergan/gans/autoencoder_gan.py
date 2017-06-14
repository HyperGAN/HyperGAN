import importlib
import json
import numpy as np
import os
import sys
import time
import uuid
import copy

from hypergan.discriminators import *
from hypergan.encoders import *
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
from .standard_gan import StandardGAN

class AutoencoderGAN(StandardGAN):
    """ 
    """

    def required(self):
        return "generator".split()

    def create(self):
        config = self.config

        self.encoder = self.create_component(config.discriminator)
        self.encoder.ops.describe("encoder")
        self.encoder.create(self.inputs.x)
        self.encoder.z = tf.zeros(0)
        self.trainer = self.create_component(config.trainer)

        StandardGAN.create(self)
        cycloss = tf.reduce_mean(tf.abs(self.inputs.x-self.generator.sample))
        self.loss.sample[1] += cycloss*10
        self.trainer.create()
        self.session.run(tf.global_variables_initializer())

 
