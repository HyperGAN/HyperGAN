import tensorflow as tf
from hypergan.multi_component import MultiComponent
from tests.mocks import MockDiscriminator, mock_gan
from hypergan.encoders.uniform_encoder import UniformEncoder
from hypergan.losses.supervised_loss import SupervisedLoss
from hypergan.losses.standard_loss import StandardLoss
import hypergan as hg

def encoder(gan):
    config = {
            "projections": ['identity', 'identity'],
            "z": 2,
            "min": 0,
            "max": 1
    }
    return UniformEncoder(gan, config)

loss_config = {
        "reduce": "reduce_mean"
}

class MockLoss:
    def __init__(self, gan, sample=None):
        self.gan = gan
        self.sample = sample

class MultiComponentTest(tf.test.TestCase):
    def test_sample(self):
        with self.test_session():
            gan = mock_gan()
            mock_sample = tf.constant(1., shape=[1,1])
            multi = MultiComponent(combine='concat',
                components=[
                    MockLoss(gan, sample=mock_sample),
                    MockLoss(gan, sample=mock_sample)
            ])

            gan.encoder = multi
            self.assertEqual(gan.ops.shape(multi.sample), [1,2])

    def test_sample_loss(self):
        with self.test_session():
            gan = mock_gan()
            ops = gan.ops
            mock_sample = tf.constant(1., shape=[1])
            multi = MultiComponent(combine='add',
                components=[
                    MockLoss(gan, sample=[mock_sample, None]),
                    MockLoss(gan, sample=[mock_sample, mock_sample])
            ])
            self.assertEqual(len(multi.sample), 2)
            self.assertEqual(ops.shape(multi.sample[0]), [1])
            self.assertEqual(ops.shape(multi.sample[1]), [1])

if __name__ == "__main__":
    tf.test.main()
