import tensorflow as tf
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class VralLoss(BaseLoss):

    def _create(self, d_real, d_fake):
        TINY = 1e-12
        config = self.config
        ops = self.discriminator.ops
        square = ops.lookup('square')

        print("Initializing vral loss from ", d_fake, d_real)
        fake_mean = config.fake_mean or 1
        target_mean = config.target_mean or 0
        nf = self.N(ops.shape(d_fake), fake_mean, 1)
        f = self.F(nf)
        fnf = f.sample

        nr = self.N(ops.shape(d_real), target_mean, 1)
        r = self.R(nr)
        rnr = r.sample

        if config.g_loss == 'l2':
            g_loss = tf.square(d_fake - target_mean)
        elif config.g_loss == 'fr_l2':
            g_loss = tf.square(f.reuse(d_fake) - r.reuse(rnr))
        elif config.g_loss == 'rr_l2':
            g_loss = tf.square(r.reuse(d_fake) - r.reuse(rnr))
        else:
            g_loss = tf.square(d_fake - target_mean)

        #if config.type == 'lsgan':
        #    g_loss += square(f.sample - target_mean)

        if config.value_function == 'l2':
            vf = 0.5*square(f.reuse(d_fake) - fnf)
            vr = 0.5*square(r.reuse(d_real) - rnr)
            d_loss = vr+vf
        elif config.value_function == 'log':
            d_loss = -tf.log(tf.nn.sigmoid(f.reuse(d_fake))+TINY) - \
                      tf.log(tf.nn.sigmoid(r.reuse(d_real))+TINY)
        else:
            vf = tf.log(tf.nn.sigmoid(f.reuse(nf))+TINY) + \
                 tf.log(1 - tf.nn.sigmoid(f.reuse(d_fake))+TINY)
            vr = tf.log(tf.nn.sigmoid(r.reuse(nr))+TINY) + \
                 tf.log(1 - tf.nn.sigmoid(r.reuse(d_real))+TINY)
            d_loss = -vr-vf


        if config.type == 'log_rr':
            d_loss -= tf.log(tf.nn.sigmoid(r.reuse(d_fake))+TINY)
            d_loss -= tf.log(1 - tf.nn.sigmoid(r.reuse(d_real))+TINY)
        elif config.type == 'log_rf':
            d_loss -= tf.log(tf.nn.sigmoid(r.reuse(d_fake))+TINY)
            d_loss -= tf.log(1 - tf.nn.sigmoid(f.reuse(d_real))+TINY)
        elif config.type == 'log_fr':
            d_loss -= tf.log(tf.nn.sigmoid(f.reuse(d_fake))+TINY)
            d_loss -= tf.log(1 - tf.nn.sigmoid(r.reuse(d_real))+TINY)
        elif config.type == 'log_ff':
            d_loss -= tf.log(tf.nn.sigmoid(f.reuse(d_fake))+TINY)
            d_loss -= tf.log(1 - tf.nn.sigmoid(f.reuse(d_real))+TINY)
        elif config.type == 'log_all':
            og = tf.log(tf.nn.sigmoid(f.reuse(d_fake))+TINY) + \
                 tf.log(1 - tf.nn.sigmoid(f.reuse(d_real))+TINY)
            og2 = tf.log(tf.nn.sigmoid(r.reuse(d_real))+TINY) + \
                 tf.log(1 - tf.nn.sigmoid(r.reuse(d_fake))+TINY)
            d_loss -= og
            d_loss -= og2

        else:
            #d_loss = 0.5*square(f.reuse(d_real) - rnr) + 0.5*square(r.reuse(d_fake) - fnf)

            d_loss += 0.5*square(r.reuse(d_real) - target_mean)
            d_loss += 0.5*square(f.reuse(d_fake) - fake_mean)
            d_loss += 0.5*square(r.reuse(d_fake) - fake_mean)
            d_loss += 0.5*square(f.reuse(d_real) - target_mean)
                     

        return [d_loss, g_loss]

    def N(self, shape, mean, stddev):
        if self.config.distribution == "uniform":
            return tf.random_uniform(shape, -1, 1) + mean
        else:
            return tf.random_normal(shape, mean, stddev)

    def F(self, d_fake):
        f_discriminator = self.gan.create_component(self.config.f_discriminator, name="F_y_g_z", reuse=self.discriminator.ops._reuse, input=d_fake)

        self.discriminator.ops.add_weights(f_discriminator.variables())
        if self.discriminator.ops._reuse:
            f_discriminator.ops.stop_reuse()

        return f_discriminator

    def R(self, d_real):
        r_discriminator = self.gan.create_component(self.config.r_discriminator, name="R_y_X", reuse=self.discriminator.ops._reuse, input=d_real)

        self.discriminator.ops.add_weights(r_discriminator.variables())
        if self.discriminator.ops._reuse:
            r_discriminator.ops.stop_reuse()

        return r_discriminator



