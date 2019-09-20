import hypergan as hg
import tensorflow as tf
from mocks import MockDiscriminator, mock_gan

class GanTest(tf.test.TestCase):
    def test_hg_gan(self):
        self.assertIs(type(mock_gan()), hg.gans.standard_gan.StandardGAN)

    def test_can_create_default(self):
        config = hg.Configuration.load('default.json')
        self.assertIs(type(mock_gan(config=config)), hg.gans.standard_gan.StandardGAN)

if __name__ == "__main__":
    tf.test.main()
