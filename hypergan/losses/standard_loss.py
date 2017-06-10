import tensorflow as tf
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class StandardLoss(BaseLoss):

    def required(self):
        return "label_smooth reduce".split()

    def _create(self, d_real, d_fake):
        ops = self.ops
        config = self.config
        gan = self.gan

        d_real = config.reduce(d_real, axis = 1)
        d_fake = config.reduce(d_fake, axis = 1)

        zeros = tf.zeros_like(d_fake)
        ones = tf.ones_like(d_fake)
        if config.improved_gan:
            g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=zeros)
            d_loss = self.sigmoid_kl_with_logits(d_real, 1.-config.label_smooth)
            #TODO missing a piece? Oops
        else:
            g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=zeros)
            d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=zeros) + \
                     tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=ones)

        return [d_loss, g_loss]

