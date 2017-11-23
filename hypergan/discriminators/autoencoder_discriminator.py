import tensorflow as tf
import hyperchamber as hc
import os
import hypergan
from hypergan.discriminators.common import *

from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
from hypergan.generators.resize_conv_generator import ResizeConvGenerator
from .base_discriminator import BaseDiscriminator

class AutoencoderDiscriminator(BaseDiscriminator):

    def build(self, net):
        config = self.config
        gan = self.gan
        ops = self.ops

        encoder = config.encoder(gan, config.encoder_options or config, name=ops.description+"autoencoder-d", input=net)
        hidden = encoder.sample
        generator = config.decoder(gan, config.decoder_options or gan.config.generator, input=encoder.sample, name=ops.description+"autoencoder-g")
        print("[autoencoder discriminator] hidden layer ", hidden)

        self.ops.add_weights(encoder.variables())
        self.ops.add_weights(generator.variables())

        print("NET - ", net, generator.sample)
        error = config.distance(net, generator.sample)

        self.reconstruction = generator.sample

        self.encoder = encoder

        return error


