import tensorflow as tf
import numpy as np
import hyperchamber as hc

from hypergan.discriminators.base_discriminator import BaseDiscriminator
from hypergan.multi_component import MultiComponent
from hypergan.encoders.uniform_encoder import UniformEncoder

TINY=1e-8
class EvolutionGenerator(BaseDiscriminator):

    def __init__(self, gan, config, **kwargs):
        self.kwargs = kwargs
        kwargs = hc.Config(kwargs)
        BaseDiscriminator.__init__(self, gan, config, name=kwargs.name, input=kwargs.input,reuse=kwargs.reuse, x=kwargs.x, g=kwargs.g)

    def build(self, net):
        gan = self.gan
        config = self.config
        self.d_variables = []

        discs = []
        parents = []
        parent_child_tuples = []

        self.kwargs["reuse"]=self.ops._reuse

        def random_t(shape):
            return UniformEncoder(gan, gan.config.encoder, output_shape=shape).sample
        def random_like(x):
            shape = self.ops.shape(x)
            return random_t(shape)

        for i in range(config.parent_count or 1):
            self.first_z = random_like(net)
            self.kwargs["input"]= self.first_z
            name=self.ops.description+"_d_"+str(i)
            self.kwargs["name"]=name
            p = config['generator_class'](gan, config, **self.kwargs)
            parents.append(p)

            for j in range(config.child_count or 3):
                name=self.ops.description+"_d_"+str(i)+str(j)
                self.kwargs["name"]=name
                self.kwargs["input"]=random_like(net)
                disc = config['generator_class'](gan, config, **self.kwargs)

                discs.append(disc)
                parent_child_tuples.append((p, disc))


        self.children = discs
        self.parents = parents
        self.parent_child_tuples = parent_child_tuples

        samples = [o.sample for o in self.parents + self.children]
        self.first_sample=self.parents[0].sample
        return tf.concat(samples, axis=0)
