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

    def trainers(self):
        trainers = []

        rms_opts = {
            'g_momentum': [0,0.1,0.01,1e-6,1e-5,1e-1,0.9,0.999, 0.5],
            'd_momentum': [0,0.1,0.01,1e-6,1e-5,1e-1,0.9,0.999, 0.5],
            'd_decay': [0.8, 0.9, 0.99,0.999,0.995,0.9999,1],
            'g_decay': [0.8, 0.9, 0.99,0.999,0.995,0.9999,1],
            'clipped_gradients': [False, 1e-2],
            'clipped_d_weights': [False, 1e-2],
            'd_learn_rate': [1e-3,1e-4,5e-4,1e-6,4e-4, 5e-5],
            'g_learn_rate': [1e-3,1e-4,5e-4,1e-6,4e-4, 5e-5]
        }

        stable_rms_opts = {
            "clipped_d_weights": 0.01,
            "clipped_gradients": False,
            "d_decay": 0.995, "d_momentum": 1e-05,
            "d_learn_rate": 0.001,
            "g_decay": 0.995,
            "g_momentum": 1e-06,
            "g_learn_rate": 0.0005,
        }

        #trainers.append(hg.trainers.rmsprop_trainer.config(**rms_opts))

        adam_opts = {}

        adam_opts = {
            'd_learn_rate': [1e-3,1e-4,5e-4,1e-2,1e-6],
            'g_learn_rate': [1e-3,1e-4,5e-4,1e-2,1e-6],
            'd_beta1': [0.9, 0.99, 0.999, 0.1, 0.01, 0.2, 1e-8],
            'd_beta2': [0.9, 0.99, 0.999, 0.1, 0.01, 0.2, 1e-8],
            'g_beta1': [0.9, 0.99, 0.999, 0.1, 0.01, 0.2, 1e-8],
            'g_beta2': [0.9, 0.99, 0.999, 0.1, 0.01, 0.2, 1e-8],
            'd_epsilon': [1e-8, 1, 0.1, 0.5],
            'g_epsilon': [1e-8, 1, 0.1, 0.5],
            'd_clipped_weights': [False, 0.01],
            'clipped_gradients': [False, 0.01]
        }

        #trainers.append(hg.trainers.adam_trainer.config(**adam_opts))

        any_opts = {}

        tftrainers = [
                tf.train.AdadeltaOptimizer,
                tf.train.AdagradOptimizer,
                tf.train.GradientDescentOptimizer,
                tf.train.AdamOptimizer,

        ]
        # TODO FtrlOptimizer
        # TODO ProximalAdagradOptimizer
        # TODO ProximalGradientDescentOptimizer

        any_opts = {
            'd_learn_rate': [1e-3,1e-4,5e-4,1e-2,1e-6],
            'g_learn_rate': [1e-3,1e-4,5e-4,1e-2,1e-6],
            'd_beta1': [0.9, 0.99, 0.999, 0.1, 0.01, 0.2, 1e-8],
            'd_beta2': [0.9, 0.99, 0.999, 0.1, 0.01, 0.2, 1e-8],
            'g_beta1': [0.9, 0.99, 0.999, 0.1, 0.01, 0.2, 1e-8],
            'g_beta2': [0.9, 0.99, 0.999, 0.1, 0.01, 0.2, 1e-8],
            'd_epsilon': [1e-8, 1, 0.1, 0.5],
            'g_epsilon': [1e-8, 1, 0.1, 0.5],
            'g_momentum': [0,0.1,0.01,1e-6,1e-5,1e-1,0.9,0.999, 0.5],
            'd_momentum': [0,0.1,0.01,1e-6,1e-5,1e-1,0.9,0.999, 0.5],
            'd_decay': [0.8, 0.9, 0.99,0.999,0.995,0.9999,1],
            'g_decay': [0.8, 0.9, 0.99,0.999,0.995,0.9999,1],
            'd_rho': [0.99,0.9,0.95,0.1,0.01,0],
            'g_rho': [0.99,0.9,0.95,0.1,0.01,0],
            'd_initial_accumulator_value': [0.99,0.9,0.95,0.1,0.01],
            'g_initial_accumulator_value': [0.99,0.9,0.95,0.1,0.01],
            'd_clipped_weights': False,
            'clipped_gradients': False,
            'd_trainer':tftrainers,
            'g_trainer':tftrainers
        }

        trainers.append(hg.trainers.proportional_control_trainer.config(**any_opts))
        #trainers.append(hg.trainers.alternating_trainer.config(**any_opts))
        return trainers
     
    def losses(self):
        losses = []

        wgan_loss_opts = {
            'reverse':[True, False],
            'reduce': [tf.reduce_mean,hg.losses.wgan_loss.linear_projection,tf.reduce_sum,tf.reduce_logsumexp],
            'gradient_penalty': list(np.arange(1, 100))
        }
        lamb_loss_opts = {
            'reverse':[True, False],
            'reduce': [tf.reduce_mean,hg.losses.wgan_loss.linear_projection,tf.reduce_sum,tf.reduce_logsumexp],
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
            'alpha':[0,1e-3,1e-2,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.999],
            'beta':[0,1e-3,1e-2,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.999]

        }
        lsgan_loss_opts = {
            'reduce': [tf.reduce_mean,hg.losses.wgan_loss.linear_projection,tf.reduce_sum,tf.reduce_logsumexp],
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
            'gradient_penalty': [False, 1, 0.1, 0.01, 0.001, 0.0001, 1e-5]
        }
        standard_loss_opts= {
            'reduce': [tf.reduce_mean,hg.losses.wgan_loss.linear_projection,tf.reduce_sum,tf.reduce_logsumexp],
        'label_smooth': list(np.linspace(0, 1, num=20))
        }
        stable_loss_opts = {
          "alpha": 0.5,
          "beta": [0.5, 0.8],
          "discriminator": None,
          "label_smooth": 0.26111111111111107,
          "labels": [[
            0,
            -1,
            -1
          ]],
          "reduce": "function:tensorflow.python.ops.math_ops.reduce_mean",
          "reverse": True
        }
        began_loss_opts = {
            'k_lambda':[0.1, 0.01, 0.001, 1e-4, 1e-5],

            'initial_k':[0],
            'reduce': [tf.reduce_mean,hg.losses.wgan_loss.linear_projection,tf.reduce_sum,tf.reduce_logsumexp, tf.argmin],
            'labels': [
                [-1, 1, 0],
                [0, 1, 1],
                [0, -1, -1],
                [1, -1, 0],
                [0, -1, 1],
                [0, 1, -1],
                [0, 0.5, -0.5],
                [0.5, -0.5, 0],
                [0.5, 0, -0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0],
                [-0.5, -0.5, 0.5],
                [-1, 1, 1],
                [1, -1, -1],
                [0, 1, 1],
                [0, -1, -1]
            ],
            'gradient_penalty': [False],
            'use_k': True,
            'gamma':list(np.linspace(0, 1, num=1000))
        }

        #losses.append([hg.losses.wgan_loss.config(**wgan_loss_opts)])
        #losses.append([hg.losses.wgan_loss.config(**wgan_loss_opts)])
        #losses.append([hg.losses.lamb_gan_loss.config(**lamb_loss_opts)])
        #losses.append([hg.losses.lamb_gan_loss.config(**stable_loss_opts)])
        #losses.append([hg.losses.lamb_gan_loss.config(**stable_loss_opts)])
        #losses.append([hg.losses.lsgan_loss.config(**lsgan_loss_opts)])
        #losses.append([hg.losses.boundary_equilibrium_loss.config(**began_loss_opts)])

        stable_loss = [{
            "create": "function:hypergan.losses.boundary_equilibrium_loss.create",
            "discriminator": None,
            "gamma": 0.4,
            "gradient_penalty": False,
            "initial_k": 0.01,
            "k_lambda": 0.002,
            "labels": [
                0.5,
                -0.5,
                -0.5
            ],
            "reduce": "function:tensorflow.python.ops.math_ops.reduce_mean",
            "reverse": False,
            "type": "lsgan",
            "use_k": True
        }]


        losses.append(stable_loss)


        #losses.append([hg.losses.wgan_loss.config(**wgan_loss_opts)])
        #losses.append([hg.losses.lamb_gan_loss.config(**lamb_loss_opts)])
        #losses.append([hg.losses.standard_gan_loss.config(**standard_loss_opts)])
        #losses.append([hg.losses.lsgan_loss.config(**lsgan_loss_opts)])

        return losses

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
            print("Set ", key, value)
        
        custom = selector.random_config()

        for key,value in custom.items():
            selected[key]=value

        
        selected['dtype']=tf.float32
        return hg.config.lookup_functions(selected)


