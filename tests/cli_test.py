import hypergan as hg
import hyperchamber as hc
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

    def test_loads_config_errors_when_empty(self):
        with self.assertRaises(ValidationException):
            gan = hg.GAN()
            args = {'load': True}
            cli = hg.CLI(gan, args)
            cli.load()
            #TODO test loading

    def test_get_dimensions(self):
        with self.test_session():
            gan = hg.GAN()
            args = hc.Config({
              "size": "4"
            })
            cli = hg.CLI(gan, args)
            self.assertEqual(cli.get_dimensions()[0], 4)
            self.assertEqual(cli.get_dimensions()[1], 4)
            self.assertEqual(cli.get_dimensions()[2], 3)

            args = hc.Config({
              "size": "8x3"
            })
            cli = hg.CLI(gan, args)
            self.assertEqual(cli.get_dimensions()[0], 8)
            self.assertEqual(cli.get_dimensions()[1], 3)
            self.assertEqual(cli.get_dimensions()[2], 3)

            args = hc.Config({
              "size": "8x3x1"
            })
            cli = hg.CLI(gan, args)
            self.assertEqual(cli.get_dimensions()[0], 8)
            self.assertEqual(cli.get_dimensions()[1], 3)
            self.assertEqual(cli.get_dimensions()[2], 1)

    def test_run(self):
        with self.test_session():
            gan = hg.GAN()
            args = hc.Config({"size": "1"})
            cli = hg.CLI(gan, args)
            cli.run()
            self.assertEqual(cli.gan, gan)

    def test_step(self):
        with self.test_session():
            gan = hg.GAN()
            args = hc.Config({"size": "1", "steps": 1, "method": "train", "save_every": -1})
            cli = hg.CLI(gan, args)
            cli.step()
            self.assertEqual(cli.gan, gan)

    def test_sample(self):
        with self.test_session():
            gan = hg.GAN()
            args = hc.Config({"size": "1", "steps": 1, "method": "train", "save_every": -1})
            cli = hg.CLI(gan, args)
            cli.sample()
            self.assertEqual(cli.gan, gan)


    def test_train(self):
        with self.test_session():
            gan = hg.GAN()
            args = hc.Config({"size": "1", "steps": 1, "method": "train", "save_every": -1})
            cli = hg.CLI(gan, args)
            cli.train()
            self.assertEqual(cli.gan, gan)

    def test_run_train(self):
        with self.test_session():
            gan = hg.GAN()
            args = hc.Config({"size": "1", "steps": 1, "method": "train"})
            cli = hg.CLI(gan, args)
            cli.run()
            self.assertEqual(cli.gan, gan)


if __name__ == "__main__":
    tf.test.main()
