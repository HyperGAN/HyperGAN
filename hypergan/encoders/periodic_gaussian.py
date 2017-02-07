import tensorflow as tf
import hyperchamber as hc
import numpy as np
from .common import *

def config():
  selector = hc.Selector()
  selector.set('create', create)
  selector.set('p', 4)

  return selector.random_config()

def create(config, gan):
  z = gan.graph.z_base
  z = periodic_triangle_waveform(z, config.p)
  return gaussian_from_uniform(config, gan, z)
