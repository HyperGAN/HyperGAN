import tensorflow as tf
import hypergan.discriminators.pyramid_discriminator as disc

from unittest.mock import MagicMock

class PyramidDiscriminatorTest(tf.test.TestCase):
    
    def testDefaultGraph(self):
        with self.test_session():
            x = tf.zeros([2,64,64,3])
            g = tf.zeros([2,64,64,3])
            base_config = MagicMock(dtype=tf.float32)
            gan = MagicMock(batch_size=2, dtype=tf.float32, config=base_config)
            config = disc.config()
            net = disc.discriminator(gan, config, x, g, [], [])
            nodes = tf.get_default_graph().as_graph_def().node
            self.assertEqual(384, len(nodes), "missing nodes") 


if __name__ == "__main__":
    tf.test.main()
