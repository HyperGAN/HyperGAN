from hypergan.samplers.base_sampler import BaseSampler
from hypergan.gan_component import ValidationException, GANComponent

import numpy as np
import time
import torch
from torch import nn

class AlignedSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.inputs = None
        self.z = self.gan.latent.next().clone()
        self.upsample = nn.Upsample((self.gan.height(),self.gan.width()), mode='bilinear')


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
            self.prompt = self.gan.text_encoder.encode_text("The color green")
            self.latent = self.gan.latent.next()
            self.latent2 = self.gan.latent.next()
            #self.labels = self.gan.labels
        b = self.z.shape[0]
        #mapping = self.gan.mapping(self.z)
        #enc = self.gan.encoder(self.inputs)
        #encoded_x2 = self.gan.generator(enc)
        #g = self.gan.generator(torch.cat([self.z, self.labels], dim=1))
        #tg = self.gan.teacher(self.z, labels=self.labels)[0]
        if hasattr(self.gan, 'upsample'):
            g = self.gan.generator(self.gan.upsample(self.inputs), context={"text": self.text})
            x_inputs = self.upsample(self.gan.upsample(self.inputs))
        else:
            g = self.gan.generator(self.latent, context={"text": self.text})
            g2 = self.gan.generator(self.latent2, context={"text": self.text})
            x_inputs = self.inputs
        #g2 = self.gan.generator(self.gan.upsample(self.inputs), context={"text": self.prompt})
        return [
        #    ('input', self.inputs),
            #('x2', encoded_x2),
            #('g2', g2),
            ('x', x_inputs),
            ('x2', self.inputs),
            ('g', g),
            ('g2', g2),
            #('g2', g2),
            #('tg', tg),
            #('g2',self.gan.generator.forward(self.gan.inputs.next(1).clone().detach(), context={"y": negy_.float().view(b,1)}))
        ]

