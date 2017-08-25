from hypergan.gan_component import GANComponent

class BaseGenerator(GANComponent):
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
            sample = gan.encoder.sample
        return self.build(sample)

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
