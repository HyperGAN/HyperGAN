import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import numpy as np

class RandomSearch:
    def __init__(self, overrides):
        self.options = {
            'trainer': self.trainers(),
            'losses':self.losses(),
            'encoders':self.encoders()
         }

        self.options = {**self.options, **overrides}

    def range(self, multiplier=1):
        return list(np.linspace(0, 1, num=1000000)*multiplier)

    def trainers(self):
        trainers = []
        any_opts = {}

        tftrainers = [
                tf.train.AdadeltaOptimizer,
                tf.train.AdagradOptimizer,
                tf.train.GradientDescentOptimizer,
                tf.train.AdamOptimizer,
                tf.train.AdagradDAOptimizer, # TODO missing param
                tf.train.MomentumOptimizer,
                tf.train.ProximalGradientDescentOptimizer,
                #tf.train.FtrlOptimizer,
                tf.train.RMSPropOptimizer,
                tf.train.ProximalAdagradOptimizer,
        ]

        selector = hc.Selector({
            'd_learn_rate': self.range(.001),
            'g_learn_rate': self.range(.001),
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
            'create': [
                hg.trainers.proportional_control_trainer.create,
                hg.trainers.alternating_trainer.create
            ],
            'run': [
                hg.trainers.proportional_control_trainer.run,
                hg.trainers.alternating_trainer.run
            ]
        })
        config = selector.random_config()


        return [config]
     
    def losses(self):
        loss_opts = {
            'reverse':[True, False],
            'reduce': [tf.reduce_mean,hg.losses.wgan_loss.linear_projection,tf.reduce_sum,tf.reduce_logsumexp],
            'gradient_penalty': False,
            'labels': [
                [-1, 1, 0],
                [0, 1, 1],
                [0, -1, -1],
                [1, -1, 0],
                [0, -1, 1],
                [0, 1, -1],
                [0, 0.5, -0.5],
                [0.5, -0.5, 0],
                [0.5, 0, -0.5]
            ],
            'alpha':self.range(),
            'beta':self.range(),
            'gamma':self.range(),
            'label_smooth': self.range(),
            'use_k': [False, True],
            'initial_k': self.range(),
            'k_lambda': self.range(),
            'type': ['wgan', 'lsgan', 'softmax'],
            'create': [
                hg.losses.boundary_equilibrium_loss.create,
                hg.losses.began_softmax_loss.create,
                hg.losses.lamb_gan_loss.create,
                hg.losses.lsgan_loss.create,
                hg.losses.standard_gan_loss.create,
                hg.losses.wgan_loss.create
            ]
        }

        config = hc.Selector(loss_opts).random_config()

        return [[config]]

    def encoders(self):
        encoders = []

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
                'z': list(np.arange(0, 100)),
                'modes': list(np.arange(2,24)),
                'projections': projections,
                'min': -1,
                'max':1,
                'create': hg.encoders.uniform_encoder.create
        }

        config = hc.Selector(encoder_opts).random_config()
        encoders.append([config])
        return encoders

    def random_config(self):
        selector = hc.Selector(self.options)
        selected = dict(selector.random_config())

        selected['dtype']=tf.float32
        return hg.config.lookup_functions(selected)


