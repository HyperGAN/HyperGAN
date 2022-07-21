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
            #self.inputs = inp['img'].clone().detach().cuda()
            self.inputs = inp.clone().detach().cuda()
            if hasattr(self.gan, 'text_encoder'):
                self.text = self.gan.text_encoder.encode_text(inp['txt'])
                texts = []
                for i in range(self.text.shape[0]):
                    texts += [self.gan.config.sampling_prompt or "The color green"]
                self.prompt = self.gan.text_encoder.encode_text(texts)
                self.latent = self.gan.latent.next()
                self.latent2 = self.gan.latent.next()
            else:
                self.text = None
                self.prompt = None
                self.latent = self.gan.encoder(inp)
                self.latent2 = self.gan.encoder(inp)
            #self.labels = self.gan.labels
        b = self.z.shape[0]
        #mapping = self.gan.mapping(self.z)
        #enc = self.gan.encoder(self.inputs)
        #encoded_x2 = self.gan.generator(enc)
        #g = self.gan.generator(torch.cat([self.z, self.labels], dim=1))
        #tg = self.gan.teacher(self.z, labels=self.labels)[0]
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
        outputs = []
        if hasattr(self.gan, 'autoencoding'):

            s0 = self.gan.encode_image(self.inputs)
            g0 = self.gan.generator(s0)
            outputs.append(('ae', g0))
            s1 = self.gan.pred(torch.cat([s0.unsqueeze(1), self.latent.unsqueeze(1)], 1))
            outputs.append(('ae', self.gan.generator(s1)))

        #g2 = self.gan.generator(self.gan.upsample(self.inputs), context={"text": self.prompt})
        return [
        #    ('input', self.inputs),
            #('x2', encoded_x2),
            #('g2', g2),
            ('x', x_inputs),
            ('g', g),
            ('g2', g2)
            #('tg', tg),
            #('g2',self.gan.generator.forward(self.gan.inputs.next(1).clone().detach(), context={"y": negy_.float().view(b,1)}))
        ]+outputs

