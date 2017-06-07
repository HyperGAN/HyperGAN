import tensorflow as tf
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class StandardLoss(BaseLoss):

    def required(self):
        return "label_smooth alpha beta".split()

    def _create(self, d_real, d_fake):
        ops = self.ops
        config = self.config
        gan = self.gan

        d_real = config.reduce(d_real, axis = 1)
        d_fake = config.reduce(d_fake, axis = 1)

        zeros = tf.zeros_like(d_fake, dtype=gan.config.dtype)
        g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=zeros)
        d_loss = self.sigmoid_kl_with_logits(d_real, 1.-config.label_smooth)

        return [d_loss, g_loss]

