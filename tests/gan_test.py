import hypergan as hg
import tensorflow as tf
from tests.mocks import MockDiscriminator, mock_gan

class GanTest(tf.test.TestCase):
    def test_hg_gan(self):
        self.assertIs(type(mock_gan()), hg.gans.standard_gan.StandardGAN)

    def test_can_create_alphagan(self):
        config = hg.Configuration.load('alpha-default.json')
        self.assertIs(type(mock_gan(config=config)), hg.gans.alpha_gan.AlphaGAN)

if __name__ == "__main__":
    tf.test.main()
