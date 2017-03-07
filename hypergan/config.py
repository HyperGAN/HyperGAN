import hyperchamber as hc
import tensorflow as tf
import importlib

from hypergan.discriminators import *
from hypergan.encoders import *
from hypergan.generators import *
from hypergan.samplers import *
from hypergan.trainers import *
from hypergan.losses import *
from hypergan.util import *
import hypergan as hg

# Below are sets of configuration options:
# Each time a new random network is started a random set of configuration variables are selected.
# This is useful for hyperparameter search.  If you want to use a specific configuration use --config

def selector(args):
    selector = hc.Selector()
    selector.set('dtype', tf.float32) #The data type to use in our GAN.  Only float32 is supported at the moment

    # Z encoder configuration
    selector.set('encoders', [[uniform_encoder.config()]])

    # Generator configuration
    selector.set("generator", [resize_conv_generator.config()])

    selector.set("trainer", rmsprop_trainer.config())

    # Discriminator configuration
    discriminators = []
    for i in range(1):
        discriminators.append(pyramid_discriminator.config(layers=5))
    selector.set("discriminators", [discriminators])

    losses = []
    for i in range(1):
        losses.append(lsgan_loss.config())
    selector.set("losses", [losses])

    return selector

def random(args):
    return selector(args).random_config()


# This looks up a function by name.   Should it be part of hyperchamber?
#TODO moveme
def get_function(name):
    if name == "function:hypergan.util.ops.prelu_internal":
        return prelu("g_")

    if not isinstance(name, str):
        return name
    namespaced_method = name.split(":")[1]
    method = namespaced_method.split(".")[-1]
    namespace = ".".join(namespaced_method.split(".")[0:-1])
    return getattr(importlib.import_module(namespace),method)

# Take a config and replace any string starting with 'function:' with a function lookup.
#TODO moveme
def lookup_functions(config):
    for key, value in config.items():
        if(isinstance(value, str) and value.startswith("function:")):
            config[key]=get_function(value)
        if(isinstance(value, list) and len(value) > 0 and isinstance(value[0],str) and value[0].startswith("function:")):
            config[key]=[get_function(v) for v in value]

    return config


