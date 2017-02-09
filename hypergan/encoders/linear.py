import tensorflow as tf
import hyperchamber as hc
import numpy as np
from .common import *

def config():
  selector = hc.Selector()
  selector.set('create', create)
  selector.set('min', -1)
  selector.set('max', 1)

  return selector.random_config()

def create(config, gan):
  return tf.random_uniform([gan.config.batch_size, gan.config.z_dimensions],config.min, config.max,dtype=gan.config.dtype)
