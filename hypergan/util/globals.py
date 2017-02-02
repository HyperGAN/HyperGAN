import tensorflow as tf
import numpy as np
from scipy.misc import imsave
tensors = {}

#TODO:  This is messy - basically just a global variable system.  Should be refactored to not use this
def set_tensor(name, tensor):
  tensors[name]=tensor

def get_tensor(name, graph=tf.get_default_graph(), isOperation=False):
    return tensors[name]# || graph.as_graph_element(name)
