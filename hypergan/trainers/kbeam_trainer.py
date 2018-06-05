import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect

from hypergan.trainers.consensus_trainer import ConsensusTrainer
from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class KBeamTrainer(BaseTrainer):

    def _create(self):
        gan = self.gan
        config = self.config

        d_vars = gan.discriminator.d_variables
        g_vars = self.g_vars
        loss = self.loss
        trainers = []

        for l, d in zip(gan.loss.losses, d_vars):
            trainers += [ConsensusTrainer(self.gan, self.config, loss=l, d_vars=d, g_vars=g_vars)]

        self.trainers = trainers
        return None, None

    def required(self):
        return "trainer learn_rate".split()

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = self.loss or gan.loss
        metrics = loss.metrics
        
        losses = [t.g_loss for t in self.trainers]
        minis = sess.run(losses)
        i=np.argmax([float(x) for x in minis])
        optimizer = self.trainers[i].optimizer

        metric_values = sess.run([optimizer] + self.output_variables(metrics), feed_dict)[1:]

        if self.current_step % 100 == 0:
            print(self.output_string(metrics) % tuple([self.current_step] + metric_values))

