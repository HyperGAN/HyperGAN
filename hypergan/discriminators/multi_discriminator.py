import tensorflow as tf
import numpy as np
import hyperchamber as hc

from hypergan.discriminators.base_discriminator import BaseDiscriminator
from hypergan.multi_component import MultiComponent

TINY=1e-8
class MultiDiscriminator(BaseDiscriminator):

    def __init__(self, gan, config, **kwargs):
        self.kwargs = kwargs
        kwargs = hc.Config(kwargs)
        BaseDiscriminator.__init__(self, gan, config, name=kwargs.name, input=kwargs.input,reuse=kwargs.reuse, x=kwargs.x, g=kwargs.g)

    """Takes multiple distributions and does an additional approximator"""
    def build(self, net):
        gan = self.gan
        config = self.config
        self.d_variables = []

        discs = []
        self.kwargs["input"]=net
        self.kwargs["reuse"]=self.ops._reuse
        for i in range(config.discriminator_count or 0):
            name=self.ops.description+"_d_"+str(i)
            self.kwargs["name"]=name
            print(">>CREATING ", i)
            disc = config['discriminator_class'](gan, config, **self.kwargs)
            self.ops.add_weights(disc.variables())
            self.d_variables += [disc.variables()]

            discs.append(disc)

        for i,dconfig in enumerate(config.discriminators):
            name=self.ops.description+"_d_"+str(i)
            self.kwargs["name"]=name
            disc = dconfig['class'](gan, dconfig, **self.kwargs)

            self.ops.add_weights(disc.variables())
            self.d_variables += [disc.variables()]
            discs.append(disc)

        combine = MultiComponent(combine=self.config.combine or "concat", components=discs)
        self.sample = combine.sample
        self.children = discs
        return self.sample


