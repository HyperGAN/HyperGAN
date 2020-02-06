from hypergan.gan_component import GANComponent
import tensorflow as tf

class BaseDiscriminator(GANComponent):
    def __init__(self, gan, config, name=None, input=None, reuse=None, features=None, weights=None, biases=None):
        GANComponent.__init__(self, gan, config)
        self.input = input
        self.name = name
        self.features = features

    def create(self, net=None):
        pass
