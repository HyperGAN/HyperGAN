import tensorflow as tf
import numpy as np
from scipy.misc import imsave
tensors = {}
def set_tensor(name, tensor):
  tensors[name]=tensor

def get_tensor(name, graph=tf.get_default_graph(), isOperation=False):
    return tensors[name]# || graph.as_graph_element(name)

def plot(config, image, file):
    """ Plot a single CIFAR image."""
    image = np.squeeze(image)
    print(file, image.shape)
    imsave(file, image)


