from hypergan.gan_component import GANComponent
import tensorflow as tf

class BaseDiscriminator(GANComponent):
    def __init__(self, gan, config, name=None, input=None, reuse=None, features=None, weights=None, biases=None):
        self.input = input
        self.name = name
        self.features = features
        GANComponent.__init__(self, gan, config, name=name, reuse=reuse, weights=weights, biases=biases)

    def create(self, net=None):
        config = self.config
        gan = self.gan

        net = net or self.input

        net = self.build(net)
        self.sample = net
        return net
