import tensorflow as tf
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class EvolutionLoss(BaseLoss):

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



        if config.mutation == 'least_squares':
            a,b,c = config.labels
            square = ops.lookup('square')
            g_loss = 0.5*square(d_fake - c)
        elif config.mutation == 'improved':
            g_loss = self.sigmoid_kl_with_logits(d_fake, generator_target_probability)
        else:
            g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=zeros) 

        if config.type == 'label_smoothing':
            generator_target_probability = config.generator_target_probability or 0.8
            label_smooth = config.label_smooth or 0.2
            d_loss = self.sigmoid_kl_with_logits(d_real, 1.-label_smooth) + \
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=zeros)
        elif config.type == 'least_squares':
            a,b,c = config.labels
            d_loss = 0.5*square(d_real - b) + 0.5*square(d_fake - a)
        else:
            d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=zeros) + \
                     tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=ones)

        return [d_loss, g_loss]

    def create(self):
        gan = self.gan
        config = self.config
        ops = self.gan.ops
        split = len(gan.generator.children)+len(gan.generator.parents)+1
        #generator structure: 
        # x, gp1, ..., gpn, gc1, ..., gcm
        d_real = self.d_real
        d_fake = self.d_fake

        net = gan.discriminator.sample

        ds = self.split_batch(net, split)
        d_real = ds[0]
        d_fake = tf.add_n(ds[1:len(gan.generator.parents)+1])/(len(gan.generator.parents))
        d_loss, _ = self._create(d_real, d_fake)

        ds = self.split_batch(net, split)
        d_real = ds[0]
        d_fake = tf.add_n(ds[1+len(gan.generator.parents):])/(len(gan.generator.children))
        _, g_loss = self._create(d_real, d_fake)
        self.children_losses = self.split_batch(g_loss, len(gan.generator.children))

        d_loss = ops.squash(d_loss, config.reduce or tf.reduce_mean) #linear doesn't work with this
        g_loss = ops.squash(g_loss, config.reduce or tf.reduce_mean)

        self.sample = [d_loss, g_loss]
        self.d_loss = d_loss
        self.g_loss = g_loss

        return self.sample


