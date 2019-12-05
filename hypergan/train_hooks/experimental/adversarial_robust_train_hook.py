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
    ops = self.gan.ops
    with tf.get_default_graph().control_dependencies([clearx, clearg]):
        self.adversarial_discriminator = gan.create_component(gan.config.discriminator, name="discriminator", input=tf.concat([gan.inputs.x+self.trainablex,gan.generator.sample+self.trainableg],axis=0), features=[gan.features], reuse=True)
        self.v = tf.gradients(self.adversarial_discriminator.sample, [self.trainablex, self.trainableg])
        self.v = [self._vlambda*v/tf.norm(v, ord=2) for v in self.v]

        self.robustness_discriminator = gan.create_component(gan.config.discriminator, name='discriminator', input=tf.concat([gan.inputs.x+self.v[0], gan.generator.sample+self.v[1]], axis=0), features=[gan.features], reuse=True)

        if config.loss_type == 'gan':
            self.loss = gan.create_component(gan.config.loss, self.robustness_discriminator).sample
            self.loss = [self._lambda * self.loss[0], self._lambda * self.loss[1]]
        elif config.loss_type == 'valp':
            # https://arxiv.org/abs/1907.05681
            dy = tf.abs(self.robustness_discriminator.sample - gan.discriminator.sample)
            dy = tf.split(dy, 2)
            dx = [tf.norm(tf.reshape(self.v[0], [self.ops.shape(self.v[0])[0], -1]), axis=1), tf.norm(tf.reshape(self.v[1], [self.ops.shape(self.v[1])[0], -1]), axis=1)]
            dx = [tf.reshape(dx[0], [self.ops.shape(dx[0])[0], 1]), tf.reshape(dx[1], [self.ops.shape(dx[1])[0], 1])]
            self.gan.add_metric('dxlp', ops.squash(dx[0]))
            self.gan.add_metric('dylp', ops.squash(dy[0]))
            print('dx', dx, dy, dy[0]/dx[0])
            lvalp = [(dy[0]/(dx[0]+1e-32) - (self.config.k or 1.0)), (dy[1]/(dx[1]+1e-32) - (self.config.k or 1.0))]
            lvalp = [self.ops.squash(lvalp[0], self.gan.loss.config.reduce or tf.reduce_mean), self.ops.squash(lvalp[1], self.gan.loss.config.reduce or tf.reduce_mean)]
            #self.gan.add_metric('lvalp', lvalp[0])

            lx = self._lambda / 2.0 * tf.square(lvalp[0])
            if config.vlambda1 != 0:
              lx += -self._lambda * lvalp[0] 
            lg = self._lambda / 2.0 * tf.square(lvalp[1])
            if config.vlambda1 != 0:
              lg += -self._lambda * lvalp[1] 
            self.loss = [lx+lg, lg]
        else:
            self.loss = tf.square(self.robustness_discriminator.sample - gan.discriminator.sample)
            self.loss = [self._lambda * self.loss[0], self._lambda * self.loss[1]]


        #self.loss = gan.create_component(gan.config.loss, self.robustness_discriminator)
        self.gan.add_metric('adl', ops.squash(self.loss[0]))
        if self.loss[1] is not None:
            self.gan.add_metric('agl', ops.squash(self.loss[1]))
        #self.gan.add_metric('vx', tf.reduce_sum(self.v[0]))
        #self.gan.add_metric('vg', tf.reduce_sum(self.v[1]))

  def variables(self):
      return [self.trainablex, self.trainableg]
  def losses(self):
    return self.loss
