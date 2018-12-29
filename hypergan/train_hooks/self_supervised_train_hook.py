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

class SelfSupervisedTrainHook(BaseTrainHook):
  """https://arxiv.org/pdf/1810.11598v1.pdf"""
  def __init__(self, gan=None, config=None, trainer=None, name="SelfSupervisedTrainHook"):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)
    g_loss = []
    d_loss = []
    if hasattr(self.gan.inputs, 'frames'):
        x = gan.x0#gan.inputs.x
        g = gan.g0#gan.generator.sample
    else:
        x = gan.inputs.x
        g = gan.generator.sample
    reuse = False
    for i in range(4):
        if gan.width() != gan.height() and i % 2 == 0:
            continue
        _x = tf.image.rot90(x, i+1)
        _g = tf.image.rot90(g, i+1)
        stacked = tf.concat([_x, _g], axis=0)
        shared = gan.create_discriminator(stacked, reuse=True).named_layers['shared']
        r = gan.create_component(config["r"], input=shared, reuse=reuse)
        reuse=True
        gan.discriminator.add_variables(r)
        gan.generator.add_variables(r)
        labels = tf.one_hot(i, 4)
        _dl = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=r.sample[0])
        _gl = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=r.sample[1])
        d_loss.append(_dl)
        g_loss.append(_gl)

    self.g_loss = (self.config.alpha or 1.0) * tf.add_n(g_loss)
    self.d_loss = (self.config.beta or 1.0) * tf.add_n(d_loss)

    self.gan.add_metric('ssgl', self.g_loss)
    self.gan.add_metric('ssdl', self.d_loss)

  def losses(self):
      return [self.d_loss, self.g_loss]

  def after_step(self, step, feed_dict):
      pass

  def before_step(self, step, feed_dict):
    if step % (self.config.step_count or 1000) != 0:
      return

