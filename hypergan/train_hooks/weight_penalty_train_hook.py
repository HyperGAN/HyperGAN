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

class WeightPenaltyTrainHook(BaseTrainHook):
  def __init__(self, gan=None, config=None, trainer=None, name="WeightPenaltyTrainHook", memory_size=2, top_k=1):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)
    d_losses = []
    weights = self.gan.weights()
    config = hc.Config(config)
    if config.only_d:
        weights = self.discriminator.weights()
    else:
        weights = self.gan.weights()
    if "l2nn" in config.constraints:
        l2nn_penalties = []
        if len(weights) > 0:
            for w in weights:
                w = tf.reshape(w, [-1, self.ops.shape(w)[-1]])
                wt = tf.transpose(w)
                wtw = tf.matmul(wt,w)
                wwt = tf.matmul(w,wt)
                def _l(m):
                    m = tf.abs(m)
                    m = tf.reduce_sum(m, axis=0,keep_dims=True)
                    m = tf.maximum(m-1, 0)
                    m = tf.reduce_max(m, axis=1,keep_dims=True)
                    return m
                l2nn_penalties.append(tf.minimum(_l(wtw), _l(wwt)))
            print('l2nn_penalty', self.config.l2nn_penalty, l2nn_penalties)
            l2nn_penalty = self.config.l2nn_penalty * tf.add_n(l2nn_penalties)
            self.add_metric('l2nn_penalty', self.gan.ops.squash(l2nn_penalty))
            d_losses.append(l2nn_penalty)
    if "ortho" in config.constraints:
        penalties = []
        for w in self.gan.weights():
            print("PENALTY", w)
            w = tf.reshape(w, [-1, self.ops.shape(w)[-1]])
            wt = tf.transpose(w)
            wtw = tf.matmul(wt,w)
            wwt = tf.matmul(w,wt)
            mwtw = tf.matmul(w, wtw)
            mwwt = tf.matmul(wt, wwt)
            def _l(w,m):
                l = tf.reduce_mean(tf.abs(w - m))
                l = self.ops.squash(l)
                return l
            penalties.append(tf.minimum(_l(w, mwtw), _l(wt, mwwt)))
        penalty = self.config.ortho_penalty * tf.add_n(penalties)
        self.add_metric('ortho_penalty', self.gan.ops.squash(penalty))
        print("PENALTY", penalty)
        penalty = tf.reshape(penalty, [1,1])
        penalty = tf.tile(penalty, [self.gan.batch_size(), 1])
        d_losses.append(penalty)


    print("D_LOSSES", d_losses)
    self.loss = tf.add_n(d_losses)

  def losses(self):
    return [self.loss, self.loss]

  def after_step(self, step, feed_dict):
    pass

  def before_step(self, step, feed_dict):
    pass
