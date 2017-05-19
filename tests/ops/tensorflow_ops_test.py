import tensorflow as tf
from hypergan.ops.tensorflow_ops import TensorflowOps

from unittest.mock import MagicMock

ops = TensorflowOps()
class TensorflowOpsTest(tf.test.TestCase):
    def testLookup(self):
        with self.test_session():
            self.assertEqual(ops.lookup('tanh'), tf.nn.tanh)

    def testDType(self):
        with self.test_session():
            self.assertEqual(ops.parse_dtype('float32'), tf.float32)

    def testShape(self):
        with self.test_session():
            self.assertEqual(ops.shape(tf.constant(1)), [])

    def testSlice(self):
        with self.test_session():
            self.assertEqual(ops.shape(ops.slice(tf.constant(1,shape=[1]), [0],[1])), [1])

    def testResize(self):
        with self.test_session():
            self.assertEqual(ops.shape(ops.resize_images(tf.constant(1, shape=[1,1,1]), [2,2], 1))[1], 2)

    def testConcat(self):
        with self.test_session():
            self.assertEqual(ops.shape(ops.concat([tf.constant(1)])), [])

    def testReshape(self):
        with self.test_session():
            self.assertEqual(ops.shape(ops.reshape(tf.constant(1), [1])), [1])

    def testLinear(self):
        with self.test_session():
            self.assertEqual(0,1)

    def testConv2d(self):
        with self.test_session():
            self.assertEqual(0,1)

    def testDeconv2d(self):
        with self.test_session():
            self.assertEqual(0,1)

    def testGenerateScope(self):
        with self.test_session():
            self.assertEqual(0,1)

if __name__ == "__main__":
    tf.test.main()
