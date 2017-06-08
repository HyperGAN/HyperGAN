from hypergan.gan_component import GANComponent
import numpy as np
import tensorflow as tf

class BaseLoss(GANComponent):
    def __init__(self, gan, config):
        GANComponent.__init__(self, gan, config)
        self.metrics = {}
        self.sample = None

    def split_batch(self, net):
        """ 
        Discriminators return stacked results (on axis 0).  
        
        This splits the results.  Returns [d_real, d_fake]
        """
        ops = self.ops
        s = ops.shape(net)
        bs = self.gan.batch_size()
        net = ops.reshape(net, [bs, -1])
        d_real = ops.slice(net, [0,0], [bs//2,-1])
        d_fake = ops.slice(net, [bs//2,0], [bs//2,-1])
        return [d_real, d_fake]

    def create(self):
        gan = self.gan
        config = self.config
        ops = self.ops

        net = gan.discriminator.sample

        d_real, d_fake = self.split_batch(net)

        d_loss, g_loss = self._create(d_real, d_fake)

        if d_loss is not None:

            if config.gradient_penalty:
                d_loss += self.gradient_penalty()
            d_loss = ops.squash(d_loss, tf.reduce_mean) #TODO linear doesn't work with this, so we cant pass config.reduce
            self.metrics['d_loss'] = d_loss

        if g_loss is not None:
            g_loss = ops.squash(g_loss, tf.reduce_mean)
            self.metrics['g_loss'] = g_loss

        self.metrics = self.metrics or sample_metrics

        self.sample = [d_loss, g_loss]

        return self.sample


    def gradient_penalty(self):
        config = self.config
        gan = self.gan
        gradient_penalty = config.gradient_penalty
        x = gan.inputs.x
        g = gan.generator.sample
        shape = [1 for t in g.get_shape()]
        shape[0] = gan.batch_size()
        uniform_noise = tf.random_uniform(shape=shape,minval=0.,maxval=1.)
        interpolates = x + (1 - uniform_noise)*g
        reused_d = gan.discriminator.reuse(interpolates)
        gradients = tf.gradients(reused_d, [interpolates])[0]
        penalty = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        penalty = tf.reduce_mean(tf.square(penalty-1.))
        return float(gradient_penalty) * penalty


    def sigmoid_kl_with_logits(self, logits, targets):
       # broadcasts the same target value across the whole batch
       # this is implemented so awkwardly because tensorflow lacks an x log x op
       assert isinstance(targets, float)
       if targets in [0., 1.]:
         entropy = 0.
       else:
         entropy = - targets * np.log(targets) - (1. - targets) * np.log(1. - targets)
         return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits) * targets) - entropy
