from hypergan.gan_component import GANComponent
import tensorflow as tf

class BaseDiscriminator(GANComponent):
    def __init__(self, gan, config, name=None, input=None, reuse=None, features=None):
        self.input = input
        self.name = name
        self.features = features
        GANComponent.__init__(self, gan, config, name=name, reuse=reuse)

    def create(self, net=None):
        config = self.config
        gan = self.gan
        ops = self.ops

        net = net or self.input

        net = self.build(net)
        self.sample = net
        return net

    def reuse(self, net=None, **opts):
        config = self.config
        gan = self.gan
        ops = self.ops

        self.ops.reuse()
        net = self.build(net, **opts)
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


