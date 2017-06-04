import hypergan as hg
import tensorflow as tf
from hypergan.gan_component import ValidationException

class CliTest(tf.test.TestCase):
    def test_cli(self):
        with self.test_session():
            gan = hg.GAN()
            args = {
            }
            cli = hg.CLI(gan, args)
            self.assertEqual(cli.gan, gan)

    def test_validate_sampler(self):
        with self.assertRaises(ValidationException):
            gan = hg.GAN()
            args = {
                    'sampler': 'nonexisting'
            }
            cli = hg.CLI(gan, args)

if __name__ == "__main__":
    tf.test.main()
