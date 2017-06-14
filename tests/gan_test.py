import hypergan as hg
import tensorflow as tf

class GanTest(tf.test.TestCase):
    def test_hg_gan(self):
        self.assertIs(hg.GAN, hg.gans.standard_gan.StandardGAN)

if __name__ == "__main__":
    tf.test.main()
