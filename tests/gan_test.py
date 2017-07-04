import hypergan as hg
import tensorflow as tf

class GanTest(tf.test.TestCase):
    def test_hg_gan(self):
        self.assertIs(type(hg.GAN()), hg.gans.standard_gan.StandardGAN)

    def test_can_create_alphagan(self):
        config = hg.Configuration.load('alpha-default.json')
        self.assertIs(type(hg.GAN(config=config)), hg.gans.alpha_gan.AlphaGAN)

if __name__ == "__main__":
    tf.test.main()
