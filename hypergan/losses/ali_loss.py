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


        if config.type == 'original':
            d_loss = -tf.log(pq)-tf.log(1-pp)
            g_loss = -tf.log(1-pq)-tf.log(pp)
        elif config.type == 'least_squares':
            a,b,c = config.labels
            square = ops.lookup('square')
            d_loss = square(d_real - b) + square(d_fake - a) - square(d_fake - c)
            g_loss = square(d_fake - c) - (square(d_real - b) + square(d_fake - a))

        elif config.type == 'wasserstein':
            d_loss = -pq+pp
            g_loss = pq-pp
        else:
            zeros = tf.zeros_like(d_fake)
            ones = tf.ones_like(d_fake)
            g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=zeros) + \
                     tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=ones)
            d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=zeros) + \
                     tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=ones)

        return [d_loss, g_loss]

