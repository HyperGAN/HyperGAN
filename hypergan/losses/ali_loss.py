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
        zeros = tf.zeros_like(d_fake)
        ones = tf.ones_like(d_fake)


        if config.type == 'original':
            d_loss = -tf.log(tf.nn.sigmoid(pq))-tf.log(1-tf.nn.sigmoid(pp))
            g_loss = -tf.log(1-tf.nn.sigmoid(pq))-tf.log(tf.nn.sigmoid(pp))
        elif config.type == 'least_squares':
            a,b,c = config.labels
            square = ops.lookup('square')
            d_loss = 0.5*square(d_real - b) + 0.5*square(d_fake - a)
            g_loss = 0.5*square(d_fake - c) + 0.5*square(d_real - a)
            #g_loss = 0.5*square(d_fake - c) - d_real

            #g_loss = 0.5*square(d_fake - c) + 0.5*(b-d_real)

            
        elif config.type == 'logistic':
            d_loss = tf.nn.softplus(-d_real) + tf.nn.softplus(d_fake)
            g_loss = tf.nn.softplus(-d_fake) + tf.nn.softplus(d_real)
        elif config.type == 'wasserstein':
            d_loss = -pq+pp
            g_loss = pq-pp
        elif config.type == 'label_smoothing':
            generator_target_probability = config.generator_target_probability or 0.8
            label_smooth = config.label_smooth or 0.2
            g_loss = self.sigmoid_kl_with_logits(d_fake, generator_target_probability) + \
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=zeros)
            d_loss = self.sigmoid_kl_with_logits(d_real, 1.-label_smooth) + \
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=zeros)
        else:
            g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=zeros) + \
                     tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=ones)
            d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=zeros) + \
                     tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=ones)

        return [d_loss, g_loss]

