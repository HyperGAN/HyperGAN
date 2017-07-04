import tensorflow as tf
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class StandardLoss(BaseLoss):

    def required(self):
        return "reduce".split()

    def _create(self, d_real, d_fake):
        ops = self.ops
        config = self.config
        gan = self.gan

        generator_target_probability = config.generator_target_probability or 0.8
        label_smooth = config.label_smooth or 0.2

        zeros = tf.zeros_like(d_fake)
        ones = tf.ones_like(d_fake)
        if config.improved:
            g_loss = self.sigmoid_kl_with_logits(d_fake, generator_target_probability)
            d_loss = self.sigmoid_kl_with_logits(d_real, 1.-label_smooth) + \
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=zeros)
        else:
            g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=zeros)
            d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=zeros) + \
                     tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=ones)

        return [d_loss, g_loss]

