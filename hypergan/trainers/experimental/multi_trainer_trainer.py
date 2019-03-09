import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class MultiTrainerTrainer(BaseTrainer):
    def __init__(self, trainers):
        self.gan = trainers[0].gan
        self.config = trainers[0].config
        self.trainers = trainers
        BaseTrainer.__init__(self, self.gan, self.config)

    def required(self):
        return []

    def _create(self):
        gan = self.gan
        config = self.config

        for i, t in enumerate(self.trainers):
            t._create()
        return None

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config

        for i, t in enumerate(self.trainers):
            t.step(feed_dict)
