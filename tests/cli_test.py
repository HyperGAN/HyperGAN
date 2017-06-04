import hypergan as hg
import tensorflow as tf

class CliTest(tf.test.TestCase):
    def test_cli(self):
        with self.test_session():
            gan = hg.GAN()
            args = {
            }
            cli = hg.CLI(gan, args)
            self.assertEqual(cli.gan, gan)

if __name__ == "__main__":
    tf.test.main()
