from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import tensorflow as tf
import hypergan.util.wavegan as wavegan
import hyperchamber as hc

TINY = 1e-12

class Graph:
    def __init__(self, gan):
        self.gan = gan

    def generator(self, inputs, reuse=False):
        config = self.gan.config
        x_dims = config.x_dims
        output_channels = config.channels
        batch_size = config.batch_size

        with(tf.variable_scope("generator", reuse=reuse)):

            z = tf.concat(1, inputs)

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
        for i in gs:
            resized = tf.image.resize_images(xs[-1],[int(xs[-1].get_shape()[1]//2),int(xs[-1].get_shape()[2]//2)], 1)
            xs.append(resized)
        xs.pop()
        gs.reverse()

        discriminators = []
        for i, discriminator in enumerate(config.discriminators):
            discriminator = hc.Config(hc.lookup_functions(discriminator))
            discriminators.append(discriminator.create(self.gan, discriminator, x, g, xs, gs,prefix="d_"+str(i)))

        def split_d(net, i):
            net = tf.slice(net, [single_batch_size*i, 0], [single_batch_size, -1])
            return net

        d_reals = [split_d(x, 0) for x in discriminators]
        d_fakes = [split_d(x, 1) for x in discriminators]
        net = tf.concat(1, discriminators)

        d_real = split_d(net, 0)
        d_fake = split_d(net, 1)

        return [d_real, d_fake, d_reals, d_fakes]


    def split_categories(layer, batch_size, categories):
        start = 0
        ret = []
        for category in categories:
            count = int(category.get_shape()[1])
            ret.append(tf.slice(layer, [0, start], [batch_size, count]))
            start += count
        return ret

    def categories_loss(categories, layer, batch_size):
        loss = 0
        def split(layer):
            start = 0
            ret = []
            for category in categories:
                count = int(category.get_shape()[1])
                ret.append(tf.slice(layer, [0, start], [batch_size, count]))
                start += count
            return ret

        for category,layer_s in zip(categories, split(layer)):
            size = int(category.get_shape()[1])
            category_prior = tf.ones([batch_size, size])*np.float32(1./size)
            logli_prior = tf.reduce_sum(tf.log(category_prior + TINY) * category, reduction_indices=1)
            layer_softmax = tf.nn.softmax(layer_s)
            logli = tf.reduce_sum(tf.log(layer_softmax+TINY)*category, reduction_indices=1)
            disc_ent = tf.reduce_mean(-logli_prior)
            disc_cross_ent =  tf.reduce_mean(-logli)

            loss += disc_ent - disc_cross_ent
        return loss

    def random_category(self, batch_size, size, dtype):
        prior = tf.ones([batch_size, size])*1./size
        dist = tf.log(prior + TINY)
        with tf.device('/cpu:0'):
            sample=tf.multinomial(dist, num_samples=1)[:, 0]
            return tf.one_hot(sample, size, dtype=dtype)


    def create_z_encoding(self):
        z_base_config = hc.Config(hc.lookup_functions(self.gan.config.z_encoder_base))
        self.gan.graph.z_base = z_base_config.create(z_base_config, self.gan)
        encoders = [self.gan.graph.z_base]
        for i, encoder in enumerate(self.gan.config.z_encoders):
            encoder = hc.Config(hc.lookup_functions(encoder))
            encoders.append(encoder.create(encoder, self.gan))

        z_encoded = tf.concat(1, encoders)
        self.gan.graph.z_encoded = z_encoded

        return z_encoded

    # Used for building the tensorflow graph with only G
    def create_generator(self, graph):
        x = graph.x
        y = graph.y
        f = graph.f
        set_tensor("x", x)
        config = self.gan.config
        set_ops_globals(config.dtype, config.batch_size)
        z_dim = int(config.z)
        
        z = self.create_z_encoding()
        

        categories = [self.random_category(config.batch_size, size, config.dtype) for size in config.categories]
        if(len(categories) > 0):
            categories_t = [tf.concat(1, categories)]
        else:
            categories_t = []


        args = [y, z]+categories_t
        g = self.generator(args)
        graph.g=g
        graph.y=y
        graph.categories=categories_t

    def create(self, graph):
        x = graph.x
        y = graph.y
        f = graph.f
        config = self.gan.config
        # This is a hack to set dtype across ops.py, since each tensorflow instruction needs a dtype argument
        # TODO refactor
        set_ops_globals(config.dtype, config.batch_size)

        batch_size = config.batch_size
        z_dim = int(config.z)

        g_losses = []
        extra_g_loss = []
        d_losses = []

        #initialize with random categories
        categories = [self.random_category(config['batch_size'], size, config['dtype']) for size in config['categories']]
        if(len(categories) > 0):
            categories_t = [tf.concat(1, categories)]
        else:
            categories_t = []

        z = self.create_z_encoding()
        # create generator
        g = self.generator([y, z]+categories_t)

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
                d_losses.append(d_loss)
            if(g_loss is not None):
                g_losses.append(g_loss)

        categories_l = None
        if config['category_loss']:

            category_layer = linear(d_last_layer, sum(config['categories']), 'v_categories',stddev=0.15)
            category_layer = batch_norm(config['batch_size'], name='v_cat_loss')(category_layer)
            category_layer = config['generator.activation'](category_layer)
            categories_l = categories_loss(categories, category_layer, config['batch_size'])
            g_losses.append(-1*config['categories_lambda']*categories_l)
            d_losses.append(-1*config['categories_lambda']*categories_l)

        g_reg_losses = [var for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if 'g_' in var.name]

        d_reg_losses = [var for var in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if 'd_' in var.name]

        extra_g_loss += g_reg_losses

        g_loss = tf.reduce_mean(tf.add_n(g_losses))
        for extra in extra_g_loss:
            g_loss += extra
        d_loss = tf.reduce_mean(tf.add_n(d_losses))
        print('d_loss', d_loss)
        #for extra in d_reg_losses:
        #    d_loss += extra
        joint_loss = tf.reduce_mean(tf.add_n(g_losses + d_losses))

        summary = tf.all_variables()
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
        graph.y=y
        graph.categories=categories_t
        graph.joint_loss=joint_loss

        g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]
        d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]

        v_vars = [var for var in tf.trainable_variables() if 'v_' in var.name]
        g_vars += v_vars
        trainer = hc.Config(hc.lookup_functions(config.trainer))
        g_optimizer, d_optimizer = trainer.create(trainer, self.gan, d_vars, g_vars)
        graph.d_optimizer=d_optimizer
        graph.g_optimizer=g_optimizer
