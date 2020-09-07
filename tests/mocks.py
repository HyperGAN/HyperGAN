import hyperchamber as hc
import hypergan as hg
import torch

from hypergan.gan_component import GANComponent

def mock_gan(batch_size=1, y=1, config=None, generator_config=None):
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
            "layers": [
              "linear 32*32*1 activation=null"
            ]

        },
        "discriminator": {
          "class": "class:hypergan.discriminators.configurable_discriminator.ConfigurableDiscriminator",
          "layers":[
            "linear 1 activation=null"
          ]

        },
        "loss": {
          "class": "function:hypergan.losses.ragan_loss.RaganLoss",
          "reduce": "reduce_mean"
        },
        "trainer": {
          "class": "function:hypergan.trainers.simultaneous_trainer.SimultaneousTrainer",
          "hooks" : [

          ],
          "optimizer": {

            "class": "function:tensorflow.python.training.adam.AdamOptimizer",
            "learn_rate": 1e-4

          }

        }
    })
    return hg.GAN(config=mock_config, inputs=MockInput(batch_size=batch_size, y=y))

class MockDiscriminator(GANComponent):
    def create(self):
        self.sample = torch.zeros([2,1], dtype=torch.float32)
        return self.sample

class MockInput:
    def batch_size(self):
        return 8

    def channels(self):
        return 1

    def width(self):
        return 32

    def height(self):
        return 32

    def next(self):
        return self.sample

    def __init__(self, batch_size=2, y=1):
        batch_size=8
        self.x = torch.zeros([batch_size,1, 32,32], dtype=torch.float32)
        self.y = torch.zeros([batch_size, y], dtype=torch.float32)
        self.sample = [self.x, self.y]

