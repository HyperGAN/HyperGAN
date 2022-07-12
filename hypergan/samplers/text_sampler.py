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

    def g(self, s0, enc_text):
        return self.gan.generator(s0.unsqueeze(1))
        #return self.gan.generator(torch.cat([enc_text.unsqueeze(1), s0.unsqueeze(1)], dim=1))
    def _sample(self):
        outputs = []
        #if self.inputs is None:
        if self.inputs is None:
            inp = self.gan.inputs.next(0)
            self.inputs = inp['img'].clone().detach().cuda()
            #if hasattr(self.gan, 'text_encoder'):
            #    self.text = self.gan.text_encoder.encode_text(inp['txt'])
            #    texts = []
            #    for i in range(self.text.shape[0]):
            #        texts += [self.gan.config.sampling_prompt or "The color green"]
            #    self.prompt = self.gan.text_encoder.encode_text(texts)
            #else:
            self.text = inp['txt']
            self.text2 = []
            for i in range(len(self.text)):
                self.text2.append(self.gan.config.sampling_prompt or "The color green")
            self.latent = self.gan.latent.next()
            self.latent2 = self.gan.latent.next()
        b = self.z.shape[0]
        if hasattr(self.gan, 's0'):
            s0 = self.gan.encode_image(self.inputs)
            print(self.text)
            enc_text = self.gan.encode_text(self.text).float()
            g0 = self.g(s0, enc_text)
            enc_text2 = self.gan.encode_text(self.text2).float()
            s0p = self.gan.pred(torch.cat([enc_text.unsqueeze(1), self.latent.unsqueeze(1)], 1)).squeeze()
            g0p = self.g(s0p, enc_text)
            s1 = self.gan.pred(torch.cat([enc_text2.unsqueeze(1), self.latent.unsqueeze(1)], 1)).squeeze()
            g1 = self.g(s1, enc_text2)
            g = g0p
            g2 = g0
            outputs.append(('g1',g1))
            x_inputs = self.inputs

        elif hasattr(self.gan, 'mapping'):
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
        ]+ outputs

