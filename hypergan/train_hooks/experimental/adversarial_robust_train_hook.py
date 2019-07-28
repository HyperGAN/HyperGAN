#From https://gist.github.com/EndingCredits/b5f35e84df10d46cfa716178d9c862a3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook

class AdversarialRobustTrainHook(BaseTrainHook):
  "AR from https://openreview.net/pdf?id=HJE6X305Fm"
  def __init__(self, gan=None, config=None, trainer=None, name="GradientPenaltyTrainHook"):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)
    if hasattr(self.gan, 'x0'):
        gan_inputs = self.gan.x0
    else:
        gan_inputs = self.gan.inputs.x
    config = hc.Config(config)

    if 'lambda' in config:
        self._lambda = self.gan.configurable_param(config['lambda'])
    else:
        self._lambda = 1.0
    self._vlambda = self.gan.configurable_param(config.vlambda or 1.0)

    self.trainablex=tf.Variable(tf.zeros_like(self.gan.inputs.x))
    self.trainableg=tf.Variable(tf.zeros_like(self.gan.inputs.x))
    clearx=tf.assign(self.trainablex, tf.zeros_like(self.gan.inputs.x))
    clearg=tf.assign(self.trainableg, tf.zeros_like(self.gan.inputs.x))
    with tf.get_default_graph().control_dependencies([clearx, clearg]):
        self.adversarial_discriminator = gan.create_component(gan.config.discriminator, name="discriminator", input=tf.concat([gan.inputs.x+self.trainablex,gan.generator.sample+self.trainableg],axis=0), features=[gan.features], reuse=True)
        self.v = tf.gradients(self.adversarial_discriminator.sample, [self.trainablex, self.trainableg])
        self.v = [self._vlambda*v/tf.norm(v, ord=2) for v in self.v]

        self.robustness_discriminator = gan.create_component(gan.config.discriminator, name='discriminator', input=tf.concat([gan.inputs.x+self.v[0], gan.generator.sample+self.v[1]], axis=0), features=[gan.features], reuse=True)

        if config.loss_type == 'gan':
            self.loss = gan.create_component(gan.config.loss, self.robustness_discriminator).sample
        else:
            self.loss = tf.square(self.robustness_discriminator.sample - gan.discriminator.sample)

        self.loss = [self._lambda * self.loss[0], self._lambda * self.loss[1]]

        #self.loss = gan.create_component(gan.config.loss, self.robustness_discriminator)
        self.gan.add_metric('adl', self.loss[0])
        self.gan.add_metric('agl', self.loss[1])
        self.gan.add_metric('vx', tf.reduce_sum(self.v[0]))
        self.gan.add_metric('vg', tf.reduce_sum(self.v[1]))

  def losses(self):
    return self.loss
