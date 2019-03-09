import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class IncrementalTrainer():
    def __init__(self, trainers, schedule):
        self.trainers = trainers
        self.schedule = schedule
        self.i = 0

    def step(self, feed_dict):
        self.i += 1

        index = 0
        threshold = 0
        for val in self.schedule:
            threshold += val
            if self.i > threshold:
                index+=1
                if len(self.trainers) <= index:
                    index =len(self.trainers) - 1

        return self.trainers[index].step(feed_dict)


