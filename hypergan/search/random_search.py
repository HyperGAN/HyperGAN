import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import numpy as np
import random

from hypergan.losses.boundary_equilibrium_loss import BoundaryEquilibriumLoss
from hypergan.losses.wasserstein_loss import WassersteinLoss
from hypergan.losses.least_squares_loss import LeastSquaresLoss
from hypergan.losses.f_divergence_loss import FDivergenceLoss
from hypergan.losses.softmax_loss import SoftmaxLoss
from hypergan.losses.standard_loss import StandardLoss
from hypergan.losses.lamb_gan_loss import LambGanLoss
from hypergan.losses.vral_loss import VralLoss

class RandomSearch:
    def __init__(self, overrides):
        self.options = {
            'discriminator': self.discriminator(),
            'generator': self.generator(),
            'trainer': self.trainer(),
            'loss':self.loss(),
            'encoder':self.encoder()
         }

        self.options = {**self.options, **overrides}

    def range(self, start=0., end=1.):
        return list(np.linspace(start, end, num=1000))

    def trainer(self):
        tftrainers = [
                tf.train.AdadeltaOptimizer,
                tf.train.AdagradOptimizer,
                tf.train.GradientDescentOptimizer,
                tf.train.AdamOptimizer,
                tf.train.MomentumOptimizer,
                tf.train.RMSPropOptimizer
        ]

        selector = hc.Selector({
            'd_learn_rate': self.range(0.0001, 0.001),
            'g_learn_rate': self.range(0.0001, 0.001),
            'd_beta1': self.range(),
            'd_beta2': self.range(),
            'g_beta1': self.range(),
            'g_beta2': self.range(),
            'd_epsilon': self.range(1e-8, 0.1),
            'g_epsilon': self.range(1e-8, 0.1),
            'g_momentum': self.range(),
            'd_momentum': self.range(),
            'd_decay': self.range(),
            'g_decay': self.range(),
            'd_rho': self.range(),
            'g_rho': self.range(),
            'd_initial_accumulator_value': self.range(),
            'g_initial_accumulator_value': self.range(),
            'd_initial_gradient_squared_accumulator_value': self.range(),
            'g_initial_gradient_squared_accumulator_value': self.range(),
            'd_initial_gradient_squared_accumulator_value': self.range(),
            'g_initial_gradient_squared_accumulator_value': self.range(),
            'd_clipped_weights': False,
            'clipped_gradients': False,
            'd_trainer':tftrainers,
            'g_trainer':tftrainers,
            "d_update_steps": [1, 2, 4, 8],
            'class': [
                #hg.trainers.proportional_control_trainer.create,
                hg.trainers.alternating_trainer.AlternatingTrainer
            ]
        })
        
        config = selector.random_config()
        config['d_trainer'] = config['g_trainer']
        return config
     
    def fc_discriminator(self):
        opts = {
          "activation": ["selu", "lrelu", "relu"],
          "layer_regularizer": [None, "layer_norm"],
          "linear_type": [None, "cosine"],
          "features": [1, 10, 100, 200, 512],
          "class": "class:hypergan.discriminators.fully_connected_discriminator.FullyConnectedDiscriminator"
        }
        return hc.Selector(opts).random_config()

    def var_loss(self):
        loss_opts = {
            'class': [
                    VralLoss
            ],
            "target_mean": [-1,-0.5,0,0.5,1],
            "fake_mean": [-1,-0.5,0,0.5,1],
            'reduce': ['reduce_mean','reduce_sum','reduce_logsumexp'],
            'type': ['log_rr', 'log_rf', 'log_fr', 'log_ff', 'log_all'],
            'value_function': ['square', 'log', 'original'],
            'g_loss': ['l2','fr_l2','rr_l2'],
            
            "r_discriminator": self.fc_discriminator()

        }
        loss_opts["f_discriminator"] = loss_opts["r_discriminator"]

        return  hc.Selector(loss_opts).random_config()

    def loss(self):
        loss_opts = {
            'class': [
                    FDivergenceLoss
            ],
            "type": ["kl","js","gan","reverse_kl","pearson","squared_hellinger", "total_variation"],
            'reduce': ['reduce_mean','reduce_sum','reduce_logsumexp']
        }

        gradient_penalty = random.choice([True, False])
        if gradient_penalty:
            loss_opts["gradient_penalty_type"]=["improved-wgan","dragan"]
            loss_opts["gradient_penalty"]=self.range(0.1, 30)

        choice = hc.Selector(loss_opts).random_config()

        if random.choice([True, False]):
            choice["regularizer"] = choice["type"]
        if random.choice([True, False]):
            choice["g_loss_type"] = choice["type"]
        return choice


    def encoder(self):
        projections = []
        projections.append([hg.encoders.uniform_encoder.identity])
        projections.append([hg.encoders.uniform_encoder.sphere])
        projections.append([hg.encoders.uniform_encoder.binary])
        projections.append([hg.encoders.uniform_encoder.modal])
        projections.append([hg.encoders.uniform_encoder.modal, hg.encoders.uniform_encoder.identity])
        projections.append([hg.encoders.uniform_encoder.modal, hg.encoders.uniform_encoder.sphere, hg.encoders.uniform_encoder.identity])
        projections.append([hg.encoders.uniform_encoder.binary, hg.encoders.uniform_encoder.sphere])
        projections.append([hg.encoders.uniform_encoder.sphere, hg.encoders.uniform_encoder.identity])
        projections.append([hg.encoders.uniform_encoder.modal, hg.encoders.uniform_encoder.sphere])
        projections.append([hg.encoders.uniform_encoder.sphere, hg.encoders.uniform_encoder.identity, hg.encoders.uniform_encoder.gaussian])
        encoder_opts = {
                'z': list(np.arange(0, 100)*2),
                'modes': list(np.arange(2,24)),
                'projections': projections,
                'min': -1,
                'max':1,
                'class': hg.encoders.uniform_encoder.UniformEncoder
        }

        return hc.Selector(encoder_opts).random_config()

    def generator(self):
        generator_opts = {
            "activation":['lrelu', 'tanh', 'selu', 'prelu', 'crelu', 'nsoftplus'],
            "final_depth":[32],
            "depth_increase":[32],
            "initializer": [None, 'random'],
            "random_stddev": list(np.linspace(0.0, 0.1, num=10000)),
            "final_activation":['lrelu', 'tanh', None],
            "block_repeat_count":[1,2,3],
            "block":[
                hg.generators.common.standard_block, 
                hg.generators.common.inception_block, 
                hg.generators.common.dense_block, 
                hg.generators.common.repeating_block
                ],
            "orthogonal_initializer_gain": list(np.linspace(0.1, 2, num=100)),
            "class":[
                hg.generators.resize_conv_generator.ResizeConvGenerator
            ]
        }

        return hc.Selector(generator_opts).random_config()

    def discriminator(self):
        discriminator_opts = {
            "activation":['relu', 'lrelu', 'tanh', 'selu', 'prelu', 'crelu'],
            "final_activation":[None],
            "block_repeat_count":[1,2,3],
            "block":[hg.discriminators.common.repeating_block,
                   hg.discriminators.common.standard_block,
                   hg.discriminators.common.strided_block
                   ],
            "depth_increase":[32],
            "extra_layers": [0, 1, 2, 3],
            "extra_layers_reduction":[1,2,4],
            "fc_layer_size":[300, 400, 500],
            "fc_layers":[0,1],
            "first_conv_size":[32],
            "layers": [3,4,5,6],
            "initial_depth": [32],
            "initializer": ['orthogonal', 'random'],
            "layer_regularizer": [None,  'layer_norm'],
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

    def random_config(self):
        return hc.Selector(self.options).random_config()
