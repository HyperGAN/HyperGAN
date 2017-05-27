import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import numpy as np

class RandomSearch:
    def __init__(self, overrides):
        self.overrides = overrides

        generators = overrides['generator']
        discriminators = overrides['discriminators']
        model = overrides['model']
        batch_size = overrides['batch_size']
        self.options = {
            'batch_size':batch_size,
            'model':model,
            'trainer': self.trainers(),
            'discriminators': discriminators,
            'generator': generators,
            'losses':self.losses(),
            'encoders':self.encoders()
         }

    def range(self):
        return list(np.linspace(0, 1, num=1000000))

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
            'd_learn_rate': self.range(),
            'g_learn_rate': self.range(),
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
                hg.losses.began_h_loss.create,
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
        projections.append([hg.encoders.uniform_encoder.modal, hg.encoders.uniform_encoder.identity])
        projections.append([hg.encoders.uniform_encoder.modal, hg.encoders.uniform_encoder.sphere, hg.encoders.uniform_encoder.identity])
        projections.append([hg.encoders.uniform_encoder.binary, hg.encoders.uniform_encoder.sphere])
        projections.append([hg.encoders.uniform_encoder.sphere, hg.encoders.uniform_encoder.identity])
        projections.append([hg.encoders.uniform_encoder.modal, hg.encoders.uniform_encoder.sphere])
        projections.append([hg.encoders.uniform_encoder.sphere, hg.encoders.uniform_encoder.identity, hg.encoders.uniform_encoder.gaussian])
        encoder_opts = {
                'z': [16],
                'modes': [2,4,8,16],
                'projections': projections
                }

        stable_encoder_opts = {
          "max": 1,
          "min": -1,
          "modes": 8,
          "projections": [[
            "function:hypergan.encoders.uniform_encoder.identity",
            "function:hypergan.encoders.uniform_encoder.modal",
             "function:hypergan.encoders.uniform_encoder.sphere"
          ]],
          "z": 16
        }


        #encoders.append([hg.encoders.uniform_encoder.config(**encoder_opts)])
        encoders.append([hg.encoders.uniform_encoder.config(**stable_encoder_opts)])
        #encoders.append([custom_encoder_config()])
        return encoders

    def random_config(self):
        selected = {}

        selector = hc.Selector()
        for key,value in self.options.items():
            selector.set(key, value)
        
        custom = selector.random_config()

        for key,value in custom.items():
            selected[key]=value

        
        selected['dtype']=tf.float32
        return hg.config.lookup_functions(selected)


