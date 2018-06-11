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
        self.hist = [0 for i in range(len(self.trainers))]
        self.tfsummary_writer = tf.summary.FileWriter('./logs/sess.graph', tf.get_default_graph())
        tf.summary.scalar("zero", tf.reduce_mean(gan.loss.losses[0].sample))
        self.tfmerge_summary = tf.summary.merge_all()
        return None, None

    def required(self):
        return "trainer learn_rate".split()

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = self.loss or gan.loss
        metrics = loss.metrics
        
        losses_g = [t.g_loss for t in self.trainers]
        losses_d = [t.d_loss for t in self.trainers]
        targets_t = losses_g+losses_d
        targets = sess.run(targets_t)
        l_g = targets[:len(targets)//2]
        l_d = targets[len(targets)//2:]

        if config.criteria == '<g':
            i=np.argmin([float(x) for x in l_g])
        elif config.criteria == '>d':
            i=np.argax([float(x) for x in l_d])
        elif config.criteria == '<d':
            i=np.argmin([float(x) for x in l_d])
        else:
            # default from paper
            i=np.argmax([float(x) for x in l_g])

        i=np.argmax([float(x) for x in l_g])
        self.hist[i]+=1
        optimizer = self.trainers[i].optimizer

        for t_t, t in zip(targets_t, targets):
            feed_dict[t_t] = t

        if config.alternating_trainer:
            _ = sess.run(self.trainers[i].g_optimizer, feed_dict)
            metric_values = sess.run([t.d_optimizer for j, t in enumerate(self.trainers)] + self.output_variables(metrics))[len(self.trainers):]
        elif config.train_all:
            metric_values = sess.run([t.optimizer for j, t in enumerate(self.trainers)] + self.output_variables(metrics), feed_dict)[len(self.trainers):]
        else:
            metric_values = sess.run([t.optimizer if j == i else t.d_optimizer for j, t in enumerate(self.trainers)] + self.output_variables(metrics), feed_dict)[len(self.trainers):]

        if self.current_step % 100 == 0:
            hist_output = "  " + "".join(["D"+str(i)+":"+str(v)+" "for i, v in enumerate(self.hist)])
            print(self.output_string(metrics) % tuple([self.current_step] + metric_values)+hist_output)
            self.hist = [0 for i in range(len(self.trainers))]

