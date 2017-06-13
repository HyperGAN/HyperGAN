from hypergan.gan_component import GANComponent

class BaseGenerator(GANComponent):
    def create(self, sample=None):
        gan = self.gan
        ops = self.ops
        ops.describe("generator")
        return self.build(sample or gan.encoder.sample)

    def layer_filter(self, net):
        ops = self.ops
        gan = self.gan
        config = self.config
        if config.layer_filter:
            fltr = config.layer_filter(gan, net)
            if fltr is not None:
                net = ops.concat(axis=3, values=[net, fltr])
        return net
