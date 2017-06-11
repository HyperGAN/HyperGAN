from hypergan.gan_component import GANComponent
import tensorflow as tf

class BaseDiscriminator(GANComponent):
    def add_noise(self, net):
        config = self.config
        if not config.noise:
            return net
        print("[discriminator] adding noise", config.noise)
        net += tf.random_normal(net.get_shape(), mean=0, stddev=config.noise, dtype=tf.float32)
        return net
