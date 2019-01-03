import tensorflow as tf
import hyperchamber as hc
import hypergan as hg

from hypergan.gan_component import GANComponent

def mock_gan(batch_size=1, y=1, config=None):
    mock_config = config or hc.Config({
        "latent": {
            "class": "function:hypergan.distributions.uniform_distribution.UniformDistribution",
            "max": 1,
            "min": -1,
            "projections": [
              "function:hypergan.distributions.uniform_distribution.identity"
            ],
            "z": 128

         },
        "generator": {
            "class": "class:hypergan.discriminators.configurable_discriminator.ConfigurableDiscriminator",
            "defaults": {
              "activation": "tanh",
              "initializer": "he_normal"
            },
            "layers": [
              "linear 32*32*1 activation=null"
            ]

        },
        "discriminator": {
          "class": "class:hypergan.discriminators.configurable_discriminator.ConfigurableDiscriminator",
          "defaults":{
            "activation": "tanh",
            "initializer": "he_normal"
          },
          "layers":[
            "linear 1 activation=null"
          ]

        },
        "loss": {
          "class": "function:hypergan.losses.ragan_loss.RaganLoss",
          "reduce": "reduce_mean"
        },
        "trainer": {
          "class": "function:hypergan.trainers.alternating_trainer.AlternatingTrainer",
          "optimizer": {

            "class": "function:tensorflow.python.training.adam.AdamOptimizer",
            "learn_rate": 1e-4

          }

        }
    })
    return hg.GAN(config=mock_config, inputs=MockInput(batch_size=batch_size, y=y))

class MockDiscriminator(GANComponent):
    def create(self):
        self.sample = tf.constant(0, shape=[2,1], dtype=tf.float32)
        return self.sample

class MockInput:
    def __init__(self, batch_size=2, y=1):
        self.x= tf.constant(10., shape=[batch_size,32,32,1], dtype=tf.float32)
        self.y= tf.constant(1., shape=[batch_size, y], dtype=tf.float32)
        self.sample = [self.x, self.y]

