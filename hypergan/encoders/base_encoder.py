from hypergan.gan_component import GANComponent

class BaseEncoder(GANComponent):
     def __init__(self, gan, config, z=None):
        GANComponent.__init__(self, gan, config)
        self.z = z

