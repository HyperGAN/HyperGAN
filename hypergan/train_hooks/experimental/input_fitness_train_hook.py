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

class InputFitnessTrainHook(BaseTrainHook):
  "Keep track of Xs with high discriminator values"
  def __init__(self, gan=None, config=None, trainer=None, name="GpSnMemoryTrainHook", memory_size=2, top_k=1):
    super().__init__(config=config, gan=gan, trainer=trainer, name=name)
    gan_inputs = self.gan.inputs.x

    self.input = tf.split(self.gan.inputs.x, self.gan.batch_size(), axis=0)
    fitness = self.gan.loss.d_real
    if self.config.abs:
        fitness = tf.abs(self.gan.loss.d_real)
    if self.config.reverse:
        fitness = -self.gan.loss.d_real
    if self.config.nabs:
        fitness = -tf.abs(self.gan.loss.d_real)
    self.d_real = tf.split(fitness, self.gan.batch_size(), axis=0)
    self.feed_input = tf.split(self.gan.feed_x, self.gan.batch_size(), axis=0)

    self.sample_batch = self.gan.set_x
    cache_count = self.gan.batch_size()
    self.cache = [tf.Variable(tf.zeros_like(self.input[i])) for i in range(len(self.input))]
    print("C ", self.cache, self.input)
    self.set_cache = [tf.assign(c, x) for c, x in zip(self.cache, self.input)]
    if self.config.skip_restore is None:
        self.restore_cache = [[tf.assign(self.gan.inputs.x[i], tf.reshape(self.cache[j], self.ops.shape(self.gan.inputs.x[i]))) for i in range(self.gan.batch_size())]  for j in range(self.gan.batch_size())]
        for i in range(self.gan.batch_size()):
            restore = []
            for j in range(self.gan.batch_size()):
                op = tf.assign(self.gan.inputs.x[i], tf.reshape(self.cache[j], self.ops.shape(self.gan.inputs.x[i])))
                restore.append(op)
            self.restore_cache.append(restore)
    self.loss = [None, None]

  def after_step(self, step, feed_dict):
    pass

  def losses(self):
      return self.loss

  def before_step(self, step, feed_dict):
    if step == 0:
        self.gan.session.run(self.sample_batch)
    def sort():
        raw_winners = np.argsort(np.array(scores).flatten())
        winners = raw_winners[:self.gan.batch_size()]

        sticky = 0
        total = 0
        cache_winners = [ scorei for scorei in winners if scorei < self.gan.batch_size()]
        losers = [ i for i in range(self.gan.batch_size()) if (i+self.gan.batch_size()) not in winners ]

        if self.config.skip_restore is None:
            for loser, cache_winner in zip(cache_winners, losers):
                self.gan.session.run(self.restore_cache[loser][cache_winner])
        #if total == sticky or sticky == 0:
        #    print("Sticky "+str(sticky) + " / "+ str(total))

    if self.config.heuristic is not None:
        count = 0
        previous_last_score = 1000
    search_steps = self.config.search_steps
    if self.config.search_steps is None:
        search_steps = 1

    if search_steps == 0:
        self.gan.session.run(self.sample_batch)

    for i in range(search_steps):
        scores = []
        scores += self.gan.session.run(self.d_real)
        self.gan.session.run(self.set_cache)
        self.gan.session.run(self.sample_batch)
        scores += self.gan.session.run(self.d_real)
        sort()
        if self.config.heuristic is not None:
            last_score = np.sort(np.array(scores).flatten())[-1]
            if last_score < previous_last_score:
                count = 0
                previous_last_score = last_score
            else:
                count += 1
                if(count > self.config.heuristic):
                    #print(i+1)
                    break

        if self.config.verify:
            sortedscores = np.sort(np.array(scores).flatten())
            newscores =np.sort(np.array(self.gan.session.run(self.d_real)).flatten())
            print(i)
            print(sortedscores[:self.gan.batch_size()].flatten())
            print(np.sort(np.array(newscores).flatten()))
            print(np.sort(newscores.flatten()) == np.array(sortedscores[:self.gan.batch_size()]).flatten())


