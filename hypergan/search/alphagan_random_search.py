import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import numpy as np

from hypergan.losses.boundary_equilibrium_loss import BoundaryEquilibriumLoss
from hypergan.losses.wasserstein_loss import WassersteinLoss
from hypergan.losses.least_squares_loss import LeastSquaresLoss
from hypergan.losses.softmax_loss import SoftmaxLoss
from hypergan.losses.standard_loss import StandardLoss
from hypergan.losses.lamb_gan_loss import LambGanLoss

from hypergan.search.random_search import RandomSearch

class AlphaGANRandomSearch(RandomSearch):
    def __init__(self, overrides):
        self.options = {
            'g_encoder': self.discriminator(),
            'z_discriminator': self.discriminator(),
            'discriminator': self.discriminator(),
            'generator': self.generator(),
            'trainer': self.trainer(),
            'loss':self.loss(),
            'encoder':self.encoder()
         }

        alpha_options = {
            'g_encoder_layers': [2,3,4,5],
            'z_discriminator_layers': [0,1,2],
            'z_discriminator_extra_layers': [0,1,2],
            'z_discriminator_extra_layers_reduction': [1,2],
            'cycloss_lambda': -1,
            'd_layer_filter': [True,False],
            'g_layer_filter': [True,False],
            'encode_layer_filter': [True, False]
        }

        alpha_config = hc.Selector(alpha_options).random_config()

        self.options['g_encoder']['layers']=alpha_config.g_encoder_layers
        self.options['z_discriminator']['layers']=alpha_config.z_discriminator_layers
        self.options['z_discriminator']['extra_layers']=alpha_config.z_discriminator_extra_layers
        self.options['z_discriminator']['extra_layers_reduction']=alpha_config.z_discriminator_extra_layers_reduction
        self.options['cycloss_lambda']=alpha_config.cycloss_lambda
        self.options["class"]="class:hypergan.gans.alpha_gan.AlphaGAN"
        self.options['d_layer_filter']=alpha_config.d_layer_filter
        self.options['g_layer_filter']=alpha_config.g_layer_filter
        self.options['encode_layer_filter']=alpha_config.encode_layer_filter
        self.options = {**self.options, **overrides}
