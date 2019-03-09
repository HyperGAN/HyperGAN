from hypergan.gan_component import GANComponent
import numpy as np
import tensorflow as tf

class BaseLoss(GANComponent):
    def __init__(self, gan, config, discriminator=None, generator=None, x=None, split=2, d_fake=None, d_real=None, reuse=False, name="BaseLoss"):
        self.sample = None
        self.ops = None
        self.reuse=reuse
        self.x = x
        self.d_fake = d_fake
        self.d_real = d_real
        self.discriminator = discriminator or gan.discriminator
        self.generator = generator
        self.split = split
        GANComponent.__init__(self, gan, config, name=name)

    def reuse(self, d_real=None, d_fake=None):
        self.discriminator.ops.reuse()
        net = self._create(d_real, d_fake)
        self.discriminator.ops.stop_reuse()
        return net


    def create(self):
        gan = self.gan
        config = self.config
        ops = self.gan.ops
        split = self.split
        d_real = self.d_real
        d_fake = self.d_fake

        d_loss = None
        g_loss = None
        if d_real is None or d_fake is None:
            # Not passed in, lets populate d_real/d_fake

            net = self.discriminator.sample

            ds = self.split_batch(net, split)
            d_real = ds[0]
            d_fake = tf.add_n(ds[1:])/(len(ds)-1)
            d_loss, g_loss = self._create(d_real, d_fake)
        else:
            d_loss, g_loss = self._create(d_real, d_fake)

        d_regularizers = []
        g_regularizers = []
        d_loss_features = d_loss
        g_loss_features = g_loss
        self.d_loss_features = d_loss_features
        self.g_loss_features = g_loss_features

        if config.random_penalty:
            gp = self.random_penalty(d_fake, d_real)
            d_regularizers.append(gp)
            self.add_metric('random_penalty', ops.squash(gp, tf.reduce_mean))

        if self.gan.config.infogan and not hasattr(self.gan, 'infogan_q'):
            sample = self.gan.generator.sample
            d = self.gan.create_component(self.gan.config.discriminator, name="discriminator", input=sample, reuse=True, features=[tf.zeros([1,16,16,256])])
            last_layer = d.controls['infogan']
            q = self.gan.create_component(self.gan.config.infogan, input=(self.gan.discriminator.controls['infogan']), name='infogan')
            self.gan.infogan_q=q
            std_cont = tf.sqrt(tf.exp(q.sample))
            true = self.gan.uniform_distribution.z
            mean = tf.reshape(q.sample, self.ops.shape(true))
            std_cont = tf.reshape(std_cont, self.ops.shape(true))
            eps = (true - mean) / (std_cont + 1e-8)
            continuous = -tf.reduce_mean( -0.5 * np.log(2*np.pi)- tf.log(std_cont+1e-8)*tf.square(eps), reduction_indices=1)
            if self.gan.config.infogan.flipped:
                continuous = -continuous

            self.metrics['cinfo']=ops.squash(continuous)
            d_regularizers.append(continuous)

        d_regularizers += self.d_regularizers()
        g_regularizers += self.g_regularizers()

        print("prereg", d_loss)
        if len(d_regularizers) > 0:
            d_loss += tf.add_n(d_regularizers)
        if len(g_regularizers) > 0:
            g_loss += tf.add_n(g_regularizers)

        d_loss = ops.squash(d_loss, config.reduce or tf.reduce_mean) #linear doesn't work with this

        # TODO: Why are we squashing before gradient penalty?
        self.add_metric('d_loss', d_loss)
        if g_loss is not None:
            g_loss = ops.squash(g_loss, config.reduce or tf.reduce_mean)
            self.add_metric('g_loss', g_loss)

        self.sample = [d_loss, g_loss]
        self.d_loss = d_loss
        self.g_loss = g_loss
        self.d_fake = d_fake
        self.d_real = d_real

        return self.sample

    def d_regularizers(self):
        return []

    def g_regularizers(self):
        return []

    def rothk_penalty(self, d_real, d_fake):
        config = self.config
        g_sample = self.gan.uniform_sample
        x = self.gan.inputs.x
        gradx = tf.gradients(d_real, [x])[0]
        gradg = tf.gradients(d_fake, [g_sample])[0]
        gradx = tf.reshape(gradx, [self.ops.shape(gradx)[0], -1])
        gradg = tf.reshape(gradg, [self.ops.shape(gradg)[0], -1])
        gradx_norm = tf.norm(gradx, axis=1, keep_dims=True)
        gradg_norm = tf.norm(gradg, axis=1, keep_dims=True)
        if int(gradx_norm.get_shape()[0]) != int(d_real.get_shape()[0]):
            print("Condensing along batch for rothk")
            gradx_norm = tf.reduce_mean(gradx_norm, axis=0)
            gradg_norm = tf.reduce_mean(gradg_norm, axis=0)
        gradx = tf.square(gradx_norm) * tf.square(1-tf.nn.sigmoid(d_real))
        gradg = tf.square(gradg_norm) * tf.square(tf.nn.sigmoid(d_fake))
        loss = gradx + gradg
        loss *= config.rothk_lambda or 1
        if config.rothk_decay:
            decay_function = config.decay_function or tf.train.exponential_decay
            decay_steps = config.decay_steps or 50000
            decay_rate = config.decay_rate or 0.9
            decay_staircase = config.decay_staircase or False
            global_step = tf.train.get_global_step()
            loss = decay_function(loss, global_step, decay_steps, decay_rate, decay_staircase)

        return loss

    def random_penalty(self, d_fake, d_real):
        config = self.config
        gan = self.gan
        ops = self.gan.ops
        gradient_penalty = config.gradient_penalty
        x = self.x 
        if x is None:
            x=gan.inputs.x
        shape = [1 for t in ops.shape(x)]
        shape[0] = gan.batch_size()
        uniform_noise = tf.random_uniform(shape=shape,minval=0.,maxval=1.)
        mask = tf.cast(tf.greater(0.5, uniform_noise), tf.float32)
        #interpolates = x * mask + g * (1-mask)
        d = d_fake *(1-mask) + d_real * mask#discriminator.reuse(interpolates)
        offset = config.random_penalty_offset or -0.8
        penalty = tf.square(d - offset)
        return penalty


    def sigmoid_kl_with_logits(self, logits, targets):
       # broadcasts the same target value across the whole batch
       # this is implemented so awkwardly because tensorflow lacks an x log x op
       assert isinstance(targets, float)
       if targets in [0., 1.]:
         entropy = 0.
       else:
         entropy = - targets * np.log(targets) - (1. - targets) * np.log(1. - targets)
         return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits) * targets) - entropy
