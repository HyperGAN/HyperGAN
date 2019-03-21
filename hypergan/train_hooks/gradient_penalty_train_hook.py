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

class GradientPenaltyTrainHook(BaseTrainHook):
  def __init__(self, gan=None, config=None, trainer=None, name="GradientPenaltyTrainHook"):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)
    if hasattr(self.gan, 'x0'):
        gan_inputs = self.gan.x0
    else:
        gan_inputs = self.gan.inputs.x

    self._lambda = self.gan.configurable_param(config['lambda'] or 1)

    if self.config.target:
        v = getattr(gan, self.config.target)
    else:
        v = gan.discriminator
    target = v.sample
    if "components" in self.config:
        target_vars = []
        for component in self.config.components:
            c = getattr(gan, component)
            target_vars += c.variables()
    else:
        target_vars = self.gan.variables()

    gd = tf.gradients(target, target_vars)
    gds = [tf.square(_gd) for _gd in gd if _gd is not None]
    if self.config.flex:
        if isinstance(self.config.flex,list):
            gds = []
            for i,flex in enumerate(self.config.flex):
                split_target = tf.split(target, len(self.config.flex))
                gd = tf.gradients(split_target, target_vars)
                fc = self.gan.configurable_param(flex)
                gds += [tf.square(tf.nn.relu(tf.abs(_gd) - fc)) for _gd in gd if _gd is not None]
        else:
            gds = [tf.square(tf.nn.relu(tf.abs(_gd) - flex)) for _gd in gd if _gd is not None]
    self.loss = tf.add_n([self._lambda * tf.reduce_mean(_r) for _r in gds])
    self.gds = gds
    self.gd = gd
    self.target_vars = target_vars
    self.target = target
    self.gan.add_metric('gp', self.loss)

  def losses(self):
    if self.config.loss == "g_loss":
        return [None, self.loss]
    else:
        return [self.loss, None]

  def after_step(self, step, feed_dict):

    if self.config.debug:
        for _v, _g in zip(self.target_vars, self.gd):
            if(_g is not None and np.mean(self.gan.session.run(_g)) > 0.1):
                print(' -> ' + _v.name,  _v, _g)
                print(" in target?", _v in self.target_vars)
                print(" in dvars? ",_v in self.gan.d_vars())
                print(" in t_dvars? ",_v in self.gan.trainable_d_vars())
                print(" in gvars? ",_v in self.gan.g_vars())
    pass

  def before_step(self, step, feed_dict):
    pass
