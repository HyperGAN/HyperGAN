from hypergan.gan_component import GANComponent

class BaseGenerator(GANComponent):

    def __init__(self, gan, config, input=None):
        GANComponent.__init__(self, gan, config)
        self.input = input

    """
        Superclass for all Generators.  Provides some common functionality.
    """
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


    def project_from_prior(self, primes, net, initial_depth, type='linear', name='prior_projection'):
        ops = self.ops
        net = ops.reshape(net, [ops.shape(net)[0], -1])
        new_shape = [ops.shape(net)[0], primes[0], primes[1], initial_depth]
        net = ops.linear(net, initial_depth*primes[0]*primes[1])
        print("projection ", net)
        net = ops.reshape(net, new_shape)
        return net


