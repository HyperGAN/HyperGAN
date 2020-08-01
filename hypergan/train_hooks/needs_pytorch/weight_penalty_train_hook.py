#From https://gist.github.com/EndingCredits/b5f35e84df10d46cfa716178d9c862a3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook

class WeightPenaltyTrainHook(BaseTrainHook):
  def __init__(self, gan=None, config=None, trainer=None, name="WeightPenaltyTrainHook", memory_size=2, top_k=1):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)
    losses = []
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
            l2nn_penalty = self.config.l2nn_penalty * tf.add_n(l2nn_penalties)
            if self.config.constrain_every is None:
                self.add_metric('l2nn_penalty', self.gan.ops.squash(l2nn_penalty))
            losses.append(l2nn_penalty)
    if "ortho" in config.constraints:
        penalties = []
        for w in weights:
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
        if self.config.constrain_every is None:
            self.add_metric('ortho_penalty', self.gan.ops.squash(penalty))
        print("PENALTY", penalty)
        penalty = tf.reshape(penalty, [1,1])
        penalty = tf.tile(penalty, [self.gan.batch_size(), 1])
        losses.append(penalty)
    if "diversity" in config.constraints:
        #Diversity regularized adversarial training
        penalties = []
        if len(weights) > 0:
            for w in weights:
                w = tf.reshape(w, [-1, self.ops.shape(w)[-1]])
                wt = tf.transpose(w)
                wt_n = tf.math.l2_normalize(wt)
                sim = tf.matmul(wt_n, tf.transpose(wt_n))
                sim = sim * (tf.ones_like(sim)-tf.eye(self.ops.shape(sim)[0]))
                mask = tf.nn.relu(tf.abs(sim)+1e-8)
                penalty = tf.reduce_sum(tf.square(sim*mask))
                penalties.append(penalty)
            full_penalty = self.config.diversity_penalty * tf.add_n(penalties)
            if self.config.constraint_every is None:
                self.add_metric('diversity_loss', self.gan.ops.squash(full_penalty))
            losses.append(full_penalty)
    print("D_L", losses)
    losses = [self.ops.squash(_l) for _l in losses]
    self.loss = tf.add_n(losses)

  def losses(self):
    return [self.loss, self.loss]

  def after_step(self, step, feed_dict):
    pass

  def before_step(self, step, feed_dict):
    pass
