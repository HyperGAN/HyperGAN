import tensorflow as tf
from hypergan.ops.tensorflow_ops import TensorflowOps

from unittest.mock import MagicMock

class TensorflowOpsTest(tf.test.TestCase):
    def testLookup(self):
        ops = TensorflowOps()
        with self.test_session():
            self.assertEqual(ops.lookup('tanh'), tf.nn.tanh)

    def testDType(self):
        ops = TensorflowOps()
        with self.test_session():
            self.assertEqual(ops.parse_dtype('float32'), tf.float32)



if __name__ == "__main__":
    tf.test.main()
