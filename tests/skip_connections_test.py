import hypergan as hg
import tensorflow as tf
from hypergan.skip_connections import SkipConnections


sc = SkipConnections()
class SkipConnectionsTest(tf.test.TestCase):
    def test_get_none(self):
        self.assertIsNone(sc.get('none'))

if __name__ == "__main__":
    tf.test.main()
