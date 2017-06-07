import tensorflow as tf
import hyperchamber as hc
import hypergan as hg

from hypergan.gan_component import GANComponent

def mock_gan(batch_size=1):
    return hg.GAN(inputs=MockInput(batch_size=batch_size))

class MockDiscriminator(GANComponent):
    def create(self):
        self.sample = tf.constant(0, shape=[2,1], dtype=tf.float32)
        return self.sample

class MockInput:
    def __init__(self, batch_size=1):
        self.x= tf.constant(10., shape=[batch_size,32,32,1], dtype=tf.float32)
        self.y= tf.constant(1., shape=[batch_size, 2], dtype=tf.float32)
        self.sample = [self.x, self.y]

