import tensorflow as tf
import hyperchamber as hc
from .common import *

def config():
  selector = hc.Selector()
  selector.set('create', create)

  return selector.random_config()

def create(config, gan):
  z = gan.graph.z_base
  return gaussian_from_uniform(config, gan, z)
