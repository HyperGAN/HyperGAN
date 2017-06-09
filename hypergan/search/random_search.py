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

class RandomSearch:
    def __init__(self, overrides):
        self.options = {
            'trainer': self.trainer(),
            'loss':self.loss(),
            'encoder':self.encoder()
         }

        self.options = {**self.options, **overrides}

    def range(self, multiplier=1):
        return list(np.linspace(0, 1, num=1000000)*multiplier)

    def trainer(self):
        tftrainers = [
                tf.train.AdadeltaOptimizer,
                tf.train.AdagradOptimizer,
                tf.train.GradientDescentOptimizer,
                tf.train.AdamOptimizer,
               # tf.train.AdagradDAOptimizer, # TODO missing param
                tf.train.MomentumOptimizer,
                #tf.train.ProximalGradientDescentOptimizer,
                #tf.train.FtrlOptimizer,
                tf.train.RMSPropOptimizer,
                #tf.train.ProximalAdagradOptimizer,
        ]

        selector = hc.Selector({
            'd_learn_rate': self.range(.01),
            'g_learn_rate': self.range(.01),
            'd_beta1': self.range(),
            'd_beta2': self.range(),
            'g_beta1': self.range(),
            'g_beta2': self.range(),
            'd_epsilon': self.range(),
            'g_epsilon': self.range(),
            'g_momentum': self.range(),
            'd_momentum': self.range(),
            'd_decay': self.range(),
            'g_decay': self.range(),
            'd_rho': self.range(),
            'g_rho': self.range(),
            'd_global_step': self.range(),
            'g_global_step': self.range(),
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
            'class': [
                #hg.trainers.proportional_control_trainer.create,
                hg.trainers.alternating_trainer.AlternatingTrainer
            ]
        })
        
        return selector.random_config()
     
    def loss(self):
        loss_opts = {
            'reverse':[True, False],
            'reduce': ['reduce_mean','reduce_sum','reduce_logsumexp'],
            'gradient_penalty': [False],
            'labels': [
                [-1, 1, 1],
                [1, -1, -1],
                [-1, 1, 1],
                [0.5, 0.5, -0.5],
                [0.5, -0.5, -0.5],
                [-0.5, 0.5, 0.5]
            ],
            'alpha':self.range(),
            'beta':self.range(),
            'gamma':self.range(),
            'label_smooth': self.range(),
            'use_k': [False, True],
            'initial_k': self.range(),
            'k_lambda': self.range(.001),
            'type': ['wgan', 'lsgan', 'softmax'],
            'class': [
                BoundaryEquilibriumLoss,
                LambGanLoss,
                LeastSquaresLoss,
                StandardLoss,
                WassersteinLoss
            ]
        }

        return  hc.Selector(loss_opts).random_config()

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

    def generator_config(self):
        generator_opts = {
            "activation":['relu', 'lrelu', 'tanh'],
            "final_depth":[32],
            "depth_increase":[32],
            "final_activation":['lrelu', 'tanh'],
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

    def discriminator_config(self):
        discriminator_opts = {
            "activation":['relu', 'lrelu', 'tanh'],
            "final_activation":['relu', 'lrelu', 'tanh'],
            "block_repeat_count":[1,2,3],
            "block":[hg.discriminators.common.repeating_block,
                   hg.discriminators.common.standard_block,
                   hg.discriminators.common.strided_block
                   ],
            "depth_increase":[32],
            "extra_layers_reduction":[1,2,4],
            "fc_layer_size":[300],
            "fc_layers":[0,1],
            "first_conv_size":[32],
            "layers": [3,4,5,6],
            "initial_depth": [32],
            "initializer": ['orthogonal', 'random'],
            "layer_regularizer": [None, 'batch_norm', 'layer_norm'],
            "noise":[False, 1e-2],
            "progressive_enhancement":[False, True],
            "orthogonal_gain": list(np.linspace(0.1, 2, num=10000)),
            "random_stddev": list(np.linspace(0.0, 0.1, num=10000)),
            "distance":[hg.discriminators.autoencoder_discriminator.l1_distance, hg.discriminators.autoencoder_discriminator.l2_distance],
            "class":[
                hg.discriminators.pyramid_discriminator.PyramidDiscriminator,
                hg.discriminators.pyramid_discriminator.AutoencoderDiscriminator
            ]
        }

        return hc.Selector(discriminator_opts).random_config()

    def random_config(self):
        return hc.Selector(self.options).random_config()
