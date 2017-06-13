from hypergan.gan_component import GANComponent
import tensorflow as tf

class BaseDiscriminator(GANComponent):
    def create(self, x=None, g=None):
        config = self.config
        gan = self.gan
        ops = self.ops

        if x is None:
            x = gan.inputs.x
        if g is None:
            g = gan.generator.sample

        x, g = self.resize(config, x, g)
        net = self.combine_filter(config, x, g)
        net = self.build(net)
        self.sample = net
        return net

    def reuse(self, x=None, g=None):
        config = self.config
        gan = self.gan
        ops = self.ops

        if x is None:
            x or gan.inputs.x
        if g is None:
            g or gan.generator.sample

        x, g = self.resize(config, x, g)
        net = self.combine_filter(config, x, g)

        self.ops.reuse()
        net = self.build(net)
        self.ops.stop_reuse()

        return net

    def add_noise(self, net):
        config = self.config
        if not config.noise:
            return net
        print("[discriminator] adding noise", config.noise)
        net += tf.random_normal(net.get_shape(), mean=0, stddev=config.noise, dtype=tf.float32)
        return net

    def resize(self, config, x, g):
        if(config.resize):
            # shave off layers >= resize 
            def should_ignore_layer(layer, resize):
                return int(layer.get_shape()[1]) > config['resize'][0] or \
                       int(layer.get_shape()[2]) > config['resize'][1]

            xs = [px for px in xs if not should_ignore_layer(px, config['resize'])]
            gs = [pg for pg in gs if not should_ignore_layer(pg, config['resize'])]

            x = tf.image.resize_images(x,config['resize'], 1)
            g = tf.image.resize_images(g,config['resize'], 1)

        else:
            return x, g

    def combine_filter(self, config, x, g):
        # TODO: This is standard optimization from improved GAN, cross-d feature
        if 'layer_filter' in config:
            g_filter = tf.concat(axis=3, values=[g, config['layer_filter'](gan, x)])
            x_filter = tf.concat(axis=3, values=[x, config['layer_filter'](gan, x)])
            net = tf.concat(axis=0, values=[x_filter,g_filter] )
        else:
            print("XG", x, g)
            net = tf.concat(axis=0, values=[x,g])
        return net

    def progressive_enhancement(self, config, net, xg):
        if 'progressive_enhancement' in config and config.progressive_enhancement and xg is not None:
            net = tf.concat(axis=3, values=[net, xg])
        return net
