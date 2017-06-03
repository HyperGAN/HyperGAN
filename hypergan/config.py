import hyperchamber as hc
import importlib

from hypergan.discriminators import *
from hypergan.encoders import *
from hypergan.generators import *
from hypergan.samplers import *
from hypergan.trainers import *
from hypergan.losses import *
import hypergan as hg

# Below are sets of configuration options:
# Each time a new random network is started a random set of configuration variables are selected.
# This is useful for hyperparameter search.  If you want to use a specific configuration use --config

def selector(args):
    selector = hc.Selector()
    selector.set('dtype', "float32") #The data type to use in our GAN.  Only float32 is supported at the moment

    # Z encoder configuration
    selector.set('encoders', [[uniform_encoder.config()]])

    # Generator configuration
    selector.set("generator", [resize_conv_generator.config()])

    selector.set("trainer", alternating_trainer.config())

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


