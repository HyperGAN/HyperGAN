import tensorflow as tf
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class AliLoss(BaseLoss):

    def required(self):
        return "reduce".split()

    def _create(self, d_real, d_fake):
        ops = self.ops
        config = self.config
        gan = self.gan

        pq = d_real
        pp = d_fake


        d_loss = -tf.log(pq)-tf.log(1-pp)
        g_loss = -tf.log(1-pq)-tf.log(pp)

        return [d_loss, g_loss]

