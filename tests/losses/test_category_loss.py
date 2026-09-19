import tensorflow as tf
import hyperchamber as hc
import hypergan as hg
import numpy as np
from hypergan.losses.category_loss import CategoryLoss
from hypergan.ops import TensorflowOps
from hypergan.multi_component import MultiComponent


from tests.mocks import MockDiscriminator, mock_gan
from unittest.mock import MagicMock

class CategoryLossTest(tf.test.TestCase):

    def test_config(self):
        pass

if __name__ == "__main__":
    tf.test.main()
