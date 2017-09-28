from hypergan.gan_component import GANComponent
import tensorflow as tf

class BaseDiscriminator(GANComponent):
    def __init__(self, gan, config, name=None, input=None, reuse=None):
        self.input = input
        self.name = name
        GANComponent.__init__(self, gan, config, name=name, reuse=reuse)

    def create(self, net=None, x=None, g=None):
        config = self.config
        gan = self.gan
        ops = self.ops

        if net is None and self.input is not None:
            net = self.input

        if net is None:
            if x is None:
                x = gan.inputs.x
            if g is None:
                g = gan.generator.sample

            x, g = self.resize(config, x, g)
            net = tf.concat(axis=0, values=[x, g])

        net = self.build(net)
        self.sample = net
        return net

    def reuse(self, net=None, x=None, g=None):
        config = self.config
        gan = self.gan
        ops = self.ops

        if net is None:
            if x is None:
                x or gan.inputs.x
            if g is None:
                g or gan.generator.sample

            x, g = self.resize(config, x, g)
            net = tf.concat(axis=0, values=[x, g])
            net = self.layer_filter(net)

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

    def layer_filter(self, net):
        config = self.config
        gan = self.gan
        ops = self.ops
        stacks = ops.shape(net)[0] // gan.batch_size()

        split_shape = ops.shape(net)
        split_shape[-1] = gan.channels()
        split_shape[0] //= stacks
        enhance = gan.skip_connections.get('progressive_enhancement', split_shape)
        concats = [net]

        if enhance is not None and stacks == 2:
            print("[discriminator] Adding layer filter with enhancement layer", enhance, split_shape)
            new_shape = [ops.shape(net)[1], ops.shape(net)[2]]
            x = self.add_noise(self.gan.inputs.x)
            x = tf.image.resize_images(x,new_shape, 1) #TODO what if the input is user defined? i.e. 2d test
            layer = tf.concat(axis=0, values=[x, enhance])
            concats.append(layer)

        if 'layer_filter' in config and config.layer_filter is not None:
            print("[discriminator] applying layer filter", config['layer_filter'])
            filters = []
            for stack in range(stacks):
                piece = tf.slice(net, [stack * gan.batch_size(), 0,0,0], [gan.batch_size(), -1, -1, -1])
                filters.append(config.layer_filter(gan, self.config, piece))
            layer = tf.concat(axis=0, values=filters)
            concats.append(layer)

        if len(concats) > 1:
            net = tf.concat(axis=3, values=concats)

        return net
