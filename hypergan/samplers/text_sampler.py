from hypergan.samplers.base_sampler import BaseSampler
from hypergan.gan_component import ValidationException, GANComponent

import numpy as np
import time
import torch
from torch import nn

class TextSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.inputs = None
        self.z = self.gan.latent.next().clone()


    def compatible_with(gan):
        if hasattr(gan, 'encoder'):
            return True
        return False

    def _sample(self):
        #if self.inputs is None:
        if self.inputs is None:
            inp = self.gan.inputs.next(0)
            self.inputs = inp['img'].clone().detach().cuda()
            self.text = self.gan.text_encoder.encode_text(inp['txt'])
            texts = []
            for i in range(self.text.shape[0]):
                texts += [self.gan.config.sampling_prompt or "The color green"]
            self.prompt = self.gan.text_encoder.encode_text(texts)
            self.latent = self.gan.latent.next()
            self.latent2 = self.gan.latent.next()
        b = self.z.shape[0]
        if hasattr(self.gan, 'mapping'):
            g = self.gan.generator(self.gan.mapping(self.latent, context={"text": self.text}))
            g2 = self.gan.generator(self.gan.mapping(self.latent2, context={"text": self.text}))
            ('g2', g2),
            x_inputs = self.inputs
        elif hasattr(self.gan, 'upsample'):
            g = self.gan.generator(self.gan.upsample(self.inputs), context={"text": self.text})
            g2 = self.gan.generator(self.gan.upsample(self.inputs), context={"text": self.prompt})
            x_inputs = self.upsample(self.gan.upsample(self.inputs))
        else:
            g = self.gan.generator(self.latent, context={"text": self.text})
            g2 = self.gan.generator(self.latent, context={"text": self.prompt})
            x_inputs = self.inputs
        return [
            ('x', x_inputs),
            ('g', g),
            ('g2', g2)
        ]

