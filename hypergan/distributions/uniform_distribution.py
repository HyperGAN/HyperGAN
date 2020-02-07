import hyperchamber as hc
import numpy as np
import torch
from .base_distribution import BaseDistribution

from ..gan_component import ValidationException
from torch.distributions import uniform

TINY=1e-12

class UniformDistribution(BaseDistribution):
    def __init__(self, gan, config):
        BaseDistribution.__init__(self, gan, config)

    def required(self):
        return "".split()

    def validate(self):
        errors = BaseDistribution.validate(self)
        #if(self.config.z is not None and int(self.config.z) % 2 != 0):
        #    errors.append("z must be a multiple of 2 (was %2d)" % self.config.z)
        return errors

    def create(self):
        gan = self.gan
        config = self.config
        projections = []
        batch_size = self.gan.batch_size()
        self.z = uniform.Uniform(torch.Tensor([-1.0]),torch.Tensor([1.0]))

    def lookup(self, projection):
        if callable(projection):
            return projection
        if projection == 'identity':
            return identity
        if projection == 'sphere':
            return sphere
        if projection == 'gaussian':
            return gaussian
        if projection == 'periodic':
            return periodic
        return self.lookup_function(projection)

    def sample(self):
        #if 'projections' in config:
        #    for projection in config.projections:
        #        projections.append(self.lookup(projection)(config, gan, self.z))
        #else:
        #        projections.append(self.z)


        return self.z.sample(torch.Size([self.gan.batch_size(), self.config.z])).view(self.gan.batch_size(), self.config.z).cuda()

def identity(config, gan, net):
    return net

def sphere(config, gan, net):
    net = gaussian(config, gan, net)
    spherenet = tf.square(net)
    if len(spherenet.get_shape()) == 2:
        spherenet = tf.reduce_sum(spherenet, 1)
        lam = tf.sqrt(spherenet+TINY)
        return net/tf.reshape(lam,[int(lam.get_shape()[0]), 1])
    else:
        spherenet = tf.reduce_sum(spherenet, 3)
        lam = tf.sqrt(spherenet+TINY)
        return net/tf.reshape(lam,[int(lam.get_shape()[0]), int(lam.get_shape()[1]), int(lam.get_shape()[2]), 1])

def modal(config, gan, net):
    net = tf.round(net*float(config.modes))/float(config.modes)
    return net

def binary(config, gan, net):
    net = tf.greater(net, 0)
    net = tf.cast(net, tf.float32)
    return net

def zero(config, gan, net):
    return tf.zeros_like(net)

def modal_gaussian(config, gan, net):
    a = modal(config, gan, net)
    b = gaussian(config, gan, net)
    return a + b * 0.1

def modal_sphere(config, gan, net):
    net = gaussian(config, gan, net)
    net = modal(config, gan, net)
    spherenet = tf.square(net)
    spherenet = tf.reduce_sum(spherenet, 1)
    lam = tf.sqrt(spherenet+TINY)
    return net/tf.reshape(lam,[int(lam.get_shape()[0]), 1])

def modal_sphere_gaussian(config, gan, net):
    net = modal_sphere(config, gan, net)
    return net + (gaussian(config, gan, net) * 0.01)

# creates normal distribution from uniform values https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
def gaussian(config, gan, net):
    z_dim = net.get_shape().as_list()[-1]
    net = (net + 1) / 2

    if len(gan.ops.shape(net)) == 4:
        za = tf.slice(net, [0,0,0,0], [gan.batch_size(), -1, -1, z_dim//2])
        zb = tf.slice(net, [0,0,0,z_dim//2], [gan.batch_size(), -1, -1, z_dim//2])
    else:
        za = tf.slice(net, [0,0], [gan.batch_size(), z_dim//2])
        zb = tf.slice(net, [0,z_dim//2], [gan.batch_size(), z_dim//2])

    pi = np.pi
    ra = tf.sqrt(-2 * tf.log(za+TINY))*tf.cos(2*pi*zb)
    rb = tf.sqrt(-2 * tf.log(za+TINY))*tf.sin(2*pi*zb)

    return tf.reshape(tf.concat(axis=len(net.get_shape())-1, values=[ra, rb]), net.get_shape())


def periodic(config, gan, net):
    return periodic_triangle_waveform(net, config.periods)

def periodic_gaussian(config, gan, net):
    net = periodic_triangle_waveform(net, config.periods)
    return gaussian(config, gan, net)

def periodic_triangle_waveform(z, p):
    return 2.0 / np.pi * tf.asin(tf.sin(2*np.pi*z/p))

def bounded(net):
    minim = -1
    maxim = 1
    return tf.minimum(tf.maximum(net, minim), maxim)
