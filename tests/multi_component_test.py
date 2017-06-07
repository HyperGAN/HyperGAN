import tensorflow as tf
from hypergan.multi_component import MultiComponent
from tests.mocks import MockDiscriminator, mock_gan
from hypergan.encoders.uniform_encoder import UniformEncoder
import hypergan as hg

def encoder(gan):
    config = {
            "projections": ['identity', 'identity'],
            "z": 2,
            "min": 0,
            "max": 1
    }
    return UniformEncoder(gan, config)

class MultiComponentTest(tf.test.TestCase):
    def test_sample(self):
        with self.test_session():
            gan = mock_gan()
            multi = MultiComponent([encoder(gan), encoder(gan)])
            gan.encoder = multi
            gan.create()
            self.assertEqual(type(multi.z), tf.Tensor)
            self.assertEqual(type(multi.sample), tf.Tensor)

if __name__ == "__main__":
    tf.test.main()
