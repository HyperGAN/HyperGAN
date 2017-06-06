import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import numpy as np

from hypergan.samplers.grid_sampler import GridSampler
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps

class GridSamplerTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            gan = hg.GAN(graph={'x' : tf.constant(1., shape=[1,4,4,1], dtype=tf.float32)})
            gan.create()

            sampler = GridSampler(gan)
            self.assertEqual(sampler.sample('/tmp/test.png')[0]['image'].shape[-1], 1)

if __name__ == "__main__":
    tf.test.main()
