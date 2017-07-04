from hypergan.gan_component import GANComponent

class BaseGenerator(GANComponent):
    def create(self, sample=None):
        gan = self.gan
        ops = self.ops
        if sample is None:
            sample = gan.encoder.sample
        return self.build(sample)

    def layer_filter(self, net):
        ops = self.ops
        gan = self.gan
        config = self.config
        if config.layer_filter:
            print("[base generator] applying layer filter", config['layer_filter'])
            fltr = config.layer_filter(gan, self.config, net)
            if fltr is not None:
                net = ops.concat(axis=3, values=[net, fltr])
        return net
