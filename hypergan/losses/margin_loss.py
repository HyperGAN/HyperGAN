import tensorflow as tf
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class MarginLoss(BaseLoss):

    def required(self):
        return "margin".split()

    def _create(self, d_real, d_fake):
        ops = self.ops
        config = self.config
        gan = self.gan
        activation = config.activation or ops.lookup('sigmoid')

        margin = config.margin

        zeros = tf.zeros_like(d_fake)

        if activation:
            d_fake = activation(d_fake)
            d_real = activation(d_real)

        g_loss = d_fake
        d_loss = d_real + tf.maximum(0.0, margin - d_fake)

        return [d_loss, g_loss]

