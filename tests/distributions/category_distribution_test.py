import tensorflow as tf
import hyperchamber as hc
import numpy as np
import hypergan as hg
from hypergan.distributions.uniform_distribution import UniformDistribution
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock
from tests.mocks import MockDiscriminator, mock_gan

from hypergan.distributions.category_distribution import CategoryDistribution

config = {
    "categories": [1,2,3],
}
class CategoriesDistributionTest(tf.test.TestCase):
    def test_categories(self):
        with self.test_session():
            gan = mock_gan()
            distribution = CategoryDistribution(gan, config)
            gan.distribution = distribution
            gan.distribution.create()
            gan.create()
            self.assertEqual(gan.ops.shape(distribution.sample)[1]//2, len(distribution.categories))

    def test_validate(self):
        with self.assertRaises(ValidationException):
            gan = mock_gan()
            CategoryDistribution(gan, {})


if __name__ == "__main__":
    tf.test.main()
