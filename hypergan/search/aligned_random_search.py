import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import numpy as np
import random

from hypergan.losses.boundary_equilibrium_loss import BoundaryEquilibriumLoss
from hypergan.losses.wasserstein_loss import WassersteinLoss
from hypergan.losses.least_squares_loss import LeastSquaresLoss
from hypergan.losses.softmax_loss import SoftmaxLoss
from hypergan.losses.standard_loss import StandardLoss
from hypergan.losses.lamb_gan_loss import LambGanLoss

from hypergan.search.random_search import RandomSearch

import hypergan as hg

class AlignedRandomSearch(RandomSearch):
    def __init__(self, overrides):
        self.options = {
            'discriminator': self.discriminator(),
            'input_encoder': self.input_encoder(),
            'generator': self.generator(),
            'trainer': self.trainer(),
            'loss':self.loss(),
            'encoder':self.encoder()
         }

        self.options['generator']['skip_linear'] = True


        self.options['cycloss_lambda'] = random.choice([0,10])

        self.options = {**self.options, **overrides}

    def input_encoder(self):
        discriminator_opts = {
            "activation":['relu', 'lrelu', 'tanh', 'selu', 'prelu', 'crelu'],
            "final_activation":['relu', 'lrelu', 'tanh', 'selu', 'prelu', 'crelu'],
            "block_repeat_count":[1,2,3],
            "block":[
                   hg.discriminators.common.standard_block,
                   hg.discriminators.common.strided_block
                   ],
            "depth_increase":[32],
            "extra_layers": [0, 1, 2],
            "extra_layers_reduction":[1,2,4],
            "fc_layer_size":[300, 400, 500],
            "fc_layers":[0],
            "first_conv_size":[32],
            "layers": [2,3,4],
            "initial_depth": [32],
            "initializer": ['orthogonal', 'random'],
            "layer_regularizer": [None, 'batch_norm', 'layer_norm'],
            "noise":[False, 1e-2],
            "progressive_enhancement":[False, True],
            "orthogonal_gain": list(np.linspace(0.1, 2, num=10000)),
            "random_stddev": list(np.linspace(0.0, 0.1, num=10000)),
            "distance":['l1_distance', 'l2_distance'],
            "class":[
                hg.discriminators.pyramid_discriminator.PyramidDiscriminator
               # hg.discriminators.autoencoder_discriminator.AutoencoderDiscriminator
            ]
        }

        return hc.Selector(discriminator_opts).random_config()


