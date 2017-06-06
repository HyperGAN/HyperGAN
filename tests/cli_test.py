import hypergan as hg
import hyperchamber as hc
import tensorflow as tf
import os
from hypergan.gan_component import ValidationException

from tests.inputs.image_loader_test import fixture_path
import shutil

def graph():
    return hc.Config({
        'x': tf.constant(10., shape=[1,32,32,1], dtype=tf.float32)
    })

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
            gan = hg.GAN(graph=graph())
            args = {'load': True, "directory": fixture_path()}
            cli = hg.CLI(gan, args)
            cli.load()
            #TODO test loading

    def test_get_dimensions(self):
        with self.test_session():
            gan = hg.GAN(graph=graph())
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
            gan = hg.GAN(graph=graph())
            args = hc.Config({"size": "1"})
            cli = hg.CLI(gan, args)
            cli.run()
            self.assertEqual(cli.gan, gan)

    def test_step(self):
        with self.test_session():
            gan = hg.GAN(graph=graph())
            gan.create()
            args = hc.Config({"size": "1", "steps": 1, "method": "train", "save_every": -1})
            cli = hg.CLI(gan, args)
            cli.step()
            self.assertEqual(cli.gan, gan)

    def test_sample(self):
        with self.test_session():
            gan = hg.GAN(graph=graph())
            args = hc.Config({"size": "1", "steps": 1, "method": "train", "save_every": -1})
            gan.create()
            cli = hg.CLI(gan, args)
            cli.sample('/tmp/test-sample.png')
            self.assertEqual(cli.gan, gan)


    def test_train(self):
        with self.test_session():
            gan = hg.GAN(graph=graph())
            args = hc.Config({"size": "1", "steps": 1, "method": "train", "save_every": -1})
            cli = hg.CLI(gan, args)
            cli.train()
            self.assertEqual(cli.gan, gan)

    def test_run_train(self):
        with self.test_session():
            gan = hg.GAN(graph=graph())
            args = hc.Config({"size": "1", "steps": 1, "method": "train"})
            cli = hg.CLI(gan, args)
            cli.run()
            self.assertEqual(cli.gan, gan)

    def test_new(self):
        with self.test_session():
            try: 
                shutil.rmtree('/tmp/hg_new')
            except Exception:
                pass
            gan = hg.GAN(graph=graph())
            args = hc.Config({"size": "1", "steps": 1, "method": "train"})
            cli = hg.CLI(gan, args)
            cli.new("/tmp/hg_new")
            self.assertTrue(os.path.isfile('/tmp/hg_new/default.json'))
            self.assertTrue(os.path.isdir('/tmp/hg_new/samples'))
            self.assertTrue(os.path.isdir('/tmp/hg_new/saves'))

    def test_safe_new(self):
        with self.test_session():
            try: 
                shutil.rmtree('/tmp/hg_new2')
            except Exception:
                pass
            gan = hg.GAN(graph=graph())
            args = hc.Config({"size": "1", "steps": 1, "method": "train"})
            cli = hg.CLI(gan, args)
            cli.new("/tmp/hg_new2")
            with self.assertRaises(ValidationException):
                cli.new("/tmp/hg_new2")


if __name__ == "__main__":
    tf.test.main()
