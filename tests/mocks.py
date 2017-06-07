import tensorflow as tf
import hyperchamber as hc
import hypergan as hg

from hypergan.gan_component import GANComponent

def mock_gan():
    return hg.GAN(inputs=MockInput())

class MockDiscriminator(GANComponent):
    def create(self):
        self.sample = tf.constant(0, shape=[2,1], dtype=tf.float32)
        return self.sample

class MockInput:
    def __init__(self):
        self.x= tf.constant(10., shape=[1,32,32,1], dtype=tf.float32)
        self.y= tf.constant(1., shape=[1, 2], dtype=tf.float32)
        self.sample = [self.x, self.y]

