import tensorflow as tf
import hyperchamber as hc
import numpy as np
import hypergan as hg
from hypergan.encoders.uniform_encoder import UniformEncoder
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock
from tests.mocks import MockDiscriminator, mock_gan

from hypergan.encoders.category_encoder import CategoryEncoder

config = {
    "categories": [1,2,3],
}
class CategoriesEncoderTest(tf.test.TestCase):
    def test_categories(self):
        with self.test_session():
            gan = mock_gan()
            encoder = CategoryEncoder(gan, config)
            gan.encoder = encoder
            gan.encoder.create()
            gan.create()
            self.assertEqual(gan.ops.shape(encoder.sample)[1]//2, len(encoder.categories))

    def test_validate(self):
        with self.assertRaises(ValidationException):
            gan = mock_gan()
            CategoryEncoder(gan, {})


if __name__ == "__main__":
    tf.test.main()
