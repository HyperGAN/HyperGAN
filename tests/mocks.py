import tensorflow as tf
import hyperchamber as hc

from hypergan.gan_component import GANComponent

class MockDiscriminator(GANComponent):
    def create(self):
        self.sample = tf.constant(0, shape=[2,1], dtype=tf.float32)
        return self.sample

def mock_graph():
    return hc.Config({
        'x': tf.constant(10., shape=[1,32,32,1], dtype=tf.float32)
    })

