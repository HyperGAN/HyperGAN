import hypergan as hg
import hyperchamber as hc
import tensorflow as tf
import os
from hypergan.gan_component import ValidationException

from tests.inputs.image_loader_test import fixture_path
from tests.mocks import MockDiscriminator, mock_gan
import shutil

from hypergan.multi_component import MultiComponent
from hypergan.losses.supervised_loss import SupervisedLoss

class CliTest(tf.test.TestCase):
    def test_cli(self):
        with self.test_session():
            gan = hg.GAN()
            args = {
            }
            cli = hg.CLI(gan, args)
            self.assertEqual(cli.gan, gan)

    def test_run(self):
        with self.test_session():
            gan = mock_gan()
            args = hc.Config({"size": "1"})
            cli = hg.CLI(gan, args)
            cli.run()
            self.assertEqual(cli.gan, gan)

    def test_step(self):
        with self.test_session():
            gan = mock_gan()
            gan.create()
            args = hc.Config({"size": "1", "steps": 1, "method": "train", "save_every": -1})
            cli = hg.CLI(gan, args)
            cli.step()
            self.assertEqual(cli.gan, gan)

    def test_sample(self):
        with self.test_session():
            gan = mock_gan()
            args = hc.Config({"size": "1", "steps": 1, "method": "train", "save_every": -1})
            gan.create()
            cli = hg.CLI(gan, args)
            cli.sample('/tmp/test-sample.png')
            self.assertEqual(cli.gan, gan)


    def test_train(self):
        with self.test_session():
            gan = mock_gan()
            args = hc.Config({"size": "1", "steps": 1, "method": "train", "save_every": -1})
            cli = hg.CLI(gan, args)
            cli.train()
            self.assertEqual(cli.gan, gan)

    def test_adds_supervised_loss(self):
        with self.test_session():
            gan = mock_gan(y=2)
            args = hc.Config({"size": "1", "steps": 1, "method": "train", "save_every": -1})
            cli = hg.CLI(gan, args)
            gan.create()
            cli.add_supervised_loss()
            self.assertEqual(type(cli.gan.loss), MultiComponent)
            self.assertEqual(type(cli.gan.loss.components[0]), SupervisedLoss)

    def test_new(self):
        with self.test_session():
            try: 
                shutil.rmtree('/tmp/hg_new')
            except Exception:
                pass
            gan = mock_gan()
            args = hc.Config({"size": "1", "steps": 1, "method": "train"})
            cli = hg.CLI(gan, args)
            cli.new("/tmp/hg_new")
            self.assertTrue(os.path.isfile('/tmp/hg_new/default.json'))

    def test_safe_new(self):
        with self.test_session():
            try: 
                shutil.rmtree('/tmp/hg_new2')
            except Exception:
                pass
            gan = mock_gan()
            args = hc.Config({"size": "1", "steps": 1, "method": "train"})
            cli = hg.CLI(gan, args)
            cli.new("/tmp/hg_new2")
            with self.assertRaises(ValidationException):
                cli.new("/tmp/hg_new2")


if __name__ == "__main__":
    tf.test.main()
