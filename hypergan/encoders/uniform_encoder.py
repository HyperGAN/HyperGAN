import tensorflow as tf
import hyperchamber as hc
import numpy as np
from .base_encoder import BaseEncoder

from ..gan_component import ValidationException

TINY=1e-12

class UniformEncoder(BaseEncoder):
    def required(self):
        return "z min max".split()

    def validate(self):
        errors = BaseEncoder.validate(self)
        if(self.config.z is not None and int(self.config.z) % 2 != 0):
            errors.append("z must be a multiple of 2 (was %2d)" % self.config.z)
        return errors

    def create(self):
        gan = self.gan
        ops = self.ops
        config = self.config
        projections = []
        batch_size = self.gan.batch_size()
        self.z = tf.random_uniform([batch_size, int(config.z)], config.min or -1, config.max or 1, dtype=ops.dtype)
        for projection in config.projections:
            projections.append(self.lookup(projection)(config, gan, self.z))
        self.sample = tf.concat(axis=1, values=projections)
        return self.sample

    def lookup(self, projection):
        if callable(projection):
            return projection
        if projection == 'identity':
            return identity
        if projection == 'sphere':
            return sphere
        if projection == 'gaussian':
            return gaussian
        print("Warning: Encoder could not lookup symbol '"+str(projection)+"'")
        return None
        

def identity(config, gan, net):
    return net

def sphere(config, gan, net):
    net = gaussian(config, gan, net)
    spherenet = tf.square(net)
    spherenet = tf.reduce_sum(spherenet, 1)
    lam = tf.sqrt(spherenet+TINY)
    return net/tf.reshape(lam,[int(lam.get_shape()[0]), 1])

def modal(config, gan, net):
    net = tf.round(net*float(config.modes))/float(config.modes)
    return net

def binary(config, gan, net):
    net = tf.greater(net, 0)
    net = tf.cast(net, tf.float32)
    return net

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
    z_dim = int(config.z)
    net = (net + 1) / 2

    za = tf.slice(net, [0,0], [gan.batch_size(), z_dim//2])
    zb = tf.slice(net, [0,z_dim//2], [gan.batch_size(), z_dim//2])

    pi = np.pi
    ra = tf.sqrt(-2 * tf.log(za+TINY))*tf.cos(2*pi*zb)
    rb = tf.sqrt(-2 * tf.log(za+TINY))*tf.sin(2*pi*zb)

    return tf.reshape(tf.concat(axis=1, values=[ra, rb]), net.get_shape())


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
