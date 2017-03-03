from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import tensorflow as tf
import hypergan.util.wavegan as wavegan
import hyperchamber as hc

TINY = 1e-12

class Graph:
    def __init__(self, gan):
        self.gan = gan

    def generator(self, z, reuse=False):
        config = self.gan.config
        x_dims = config.x_dims
        output_channels = config.channels
        batch_size = config.batch_size

        with(tf.variable_scope("generator", reuse=reuse)):

            if 'y' in self.gan.graph:
                z = tf.concat(axis=1, values=[z, self.gan.graph.y])

            generator = hc.Config(hc.lookup_functions(config.generator))
            nets = generator.create(generator, self.gan, z)

            return nets

    def discriminator(self, x, f,z,g,gz):
        config = self.gan.config
        graph = self.gan.graph
        batch_size = config.batch_size*2
        single_batch_size = config.batch_size
        channels = config.channels
        # combine to one batch, per Ian's "Improved GAN"
        xs = [x]
        gs = g
        graph.xs=xs
        graph.gs=gs
        g = g[-1]
        if len(gs) > 1:
            for i in gs:
                resized = tf.image.resize_images(xs[-1],[int(xs[-1].get_shape()[1]//2),int(xs[-1].get_shape()[2]//2)], 1)
                xs.append(resized)
            xs.pop()
            gs.reverse()

        discriminators = []
        for i, discriminator in enumerate(config.discriminators):
            discriminator = hc.Config(hc.lookup_functions(discriminator))
            with(tf.variable_scope("discriminator")):
                discriminators.append(discriminator.create(self.gan, discriminator, x, g, xs, gs,prefix="d_"+str(i)))

        def split_d(net, i):
            net = tf.slice(net, [single_batch_size*i, 0], [single_batch_size, -1])
            return net

        d_reals = [split_d(x, 0) for x in discriminators]
        d_fakes = [split_d(x, 1) for x in discriminators]
        net = tf.concat(axis=1, values=discriminators)

        d_real = split_d(net, 0)
        d_fake = split_d(net, 1)

        return [d_real, d_fake, d_reals, d_fakes]

    def create_z_encoding(self):
        self.gan.graph.z = []
        encoders = []
        for i, encoder in enumerate(self.gan.config.encoders):
            encoder = hc.Config(hc.lookup_functions(encoder))
            zs, z_base = encoder.create(encoder, self.gan)
            encoders.append(zs)
            self.gan.graph.z.append(z_base)

        z_encoded = tf.concat(axis=1, values=encoders)
        self.gan.graph.z_encoded = z_encoded

        return z_encoded

    # Used for building the tensorflow graph with only G
    def create_generator(self, graph):
        x = graph.x
        f = graph.f
        set_tensor("x", x)
        config = self.gan.config
        set_ops_globals(config.dtype, config.batch_size)
        
        z = self.create_z_encoding()
        
        graph.g = self.generator(args)

    def create(self, graph):
        x = graph.x
        f = graph.f
        config = self.gan.config
        # This is a hack to set dtype across ops.py, since each tensorflow instruction needs a dtype argument
        # TODO refactor
        set_ops_globals(config.dtype, config.batch_size)

        batch_size = config.batch_size

        g_losses = []
        extra_g_loss = []
        d_losses = []

        z = self.create_z_encoding()
        # create generator
        g = self.generator(z)

        g_sample = g

        d_real, d_fake, d_reals, d_fakes = self.discriminator(x, f, None, g, z)

        self.gan.graph.d_real = d_real
        self.gan.graph.d_fake = d_fake
        self.gan.graph.d_reals = d_reals
        self.gan.graph.d_fakes = d_fakes

        for i, loss in enumerate(config.losses):
            loss = hc.Config(hc.lookup_functions(loss))
            d_loss, g_loss = loss.create(loss, self.gan)
            if(d_loss is not None):
                d_losses.append(tf.squeeze(d_loss))
            if(g_loss is not None):
                g_losses.append(tf.squeeze(g_loss))

        g_reg_losses = [var for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if 'g_' in var.name]

        d_reg_losses = [var for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if 'd_' in var.name]

        extra_g_loss += g_reg_losses

        g_loss = tf.reduce_mean(tf.add_n(g_losses))
        for extra in extra_g_loss:
            g_loss += extra

        d_loss = tf.reduce_mean(tf.add_n(d_losses))
        #for extra in d_reg_losses:
        #    d_loss += extra
        joint_loss = tf.reduce_mean(tf.add_n(g_losses + d_losses))

        summary = tf.global_variables()
        def summary_reduce(s):
            if(len(s.get_shape())==0):
                return s
            while(len(s.get_shape())>1):
                s=tf.reduce_mean(s,1)
                #s=tf.squeeze(s)
            return tf.reduce_mean(s,0)

        summary = [(s.get_shape(), s.name, s.dtype, summary_reduce(s)) for s in summary]

        graph.d_loss=d_loss
        graph.d_log=-tf.log(tf.abs(d_loss+TINY))
        graph.f=f
        graph.g=g_sample
        graph.g_loss=g_loss
        graph.hc_summary=summary
        graph.joint_loss=joint_loss

        g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]
        d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]

        v_vars = [var for var in tf.trainable_variables() if 'v_' in var.name]
        g_vars += v_vars
        trainer = hc.Config(hc.lookup_functions(config.trainer))
        g_optimizer, d_optimizer = trainer.create(trainer, self.gan, d_vars, g_vars)
        graph.d_optimizer=d_optimizer
        graph.g_optimizer=g_optimizer
