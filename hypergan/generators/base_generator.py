from hypergan.gan_component import GANComponent

class BaseGenerator(GANComponent):

    def __init__(self, gan, config, name=None, input=None, reuse=False):
        self.input = input
        self.name = name

        GANComponent.__init__(self, gan, config, name=name, reuse=reuse)

    """
        Superclass for all Generators.  Provides some common functionality.
    """
    def create(self, sample=None):
        """
            Creates new weights for `sample`.  Defaults to `gan.encoder.sample`
        """
        gan = self.gan
        ops = self.ops
        if sample is None:
            sample = self.input
        if sample is None:
            sample = gan.encoder.sample
        self.sample = self.build(sample)
        return self.sample

    def add_progressive_enhancement(self, net):
        ops = self.ops
        gan = self.gan
        config = self.config
        if config.progressive_enhancement:
            split = ops.slice(net, [0, 0, 0, 0], [-1, -1, -1, gan.channels()])
            if config.final_activation:
                split = config.final_activation(split)
            print("[generator] adding progressive enhancement", split)
            gan.skip_connections.set('progressive_enhancement', split)


    def layer_filter(self, net):
        """
            If a layer filter is defined, apply it.  Layer filters allow for adding information
            to every layer of the network.
        """
        ops = self.ops
        gan = self.gan
        config = self.config
        if config.layer_filter:
            print("[base generator] applying layer filter", config['layer_filter'])
            fltr = config.layer_filter(gan, self.config, net)
            if fltr is not None:
                net = ops.concat(axis=3, values=[net, fltr])
        return net

    def project_from_prior(self, primes, net, initial_depth, type='linear', name='prior_projection'):
        ops = self.ops
        net = ops.reshape(net, [ops.shape(net)[0], -1])
        new_shape = [ops.shape(net)[0], primes[0], primes[1], initial_depth]
        net = ops.linear(net, initial_depth*primes[0]*primes[1])
        print("projection ", net)
        net = ops.reshape(net, new_shape)
        return net


