from .base_generator import BaseGenerator
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.generators import legacy
import numpy as np
import torch
import torch.nn as nn
import sys

class StyleganXLGeneratorInput:
    def to(self, device):
        return StyleganXLGeneratorInput(self.config, device=device)

    def batch_size(self):
        return self.config.batch_size

    def width(self):
        return self.config.width

    def height(self):
        return self.config.height

    def channels(self):
        return self.config.channels

    def next(self, index=0, gan=None):
        if self.config.iid:
            z= gan.latent.next()
        else:
            z = gan.augmented_latent
        self.gan = gan
        device = self.gan.device
        batch_sz = self.gan.batch_size()
        G = self._generator
        w_avg = G.mapping.w_avg.unsqueeze(0)

        w = G.mapping(z, None)

        w_avg = w_avg.unsqueeze(1).repeat(1, G.mapping.num_ws, 1)
        truncation_psi = 1.0
        w = w_avg + (w - w_avg) * truncation_psi
        if self.config.mapping:
            return w

    def __init__(self, config, device=None):
        self.config = config
        sys.path.append('/ml2/trained/stylegan_xl')
        with open(self.config.pkl, "rb") as f:
            G = legacy.load_network_pkl(f)['G_ema']
            self._generator = G.eval().requires_grad_(True).to(device).cuda()


class StyleganXLGenerator(BaseGenerator):
    def create(self):
        sys.path.append('/ml2/trained/stylegan_xl')
        with open(self.config.pkl, "rb") as f:
            G = legacy.load_network_pkl(f)['G_ema']
            self._generator = G.eval().requires_grad_(True).to(self.gan.device)
        if self.config.pre:
            defn = self.config.pre.copy()
            klass = GANComponent.lookup_function(None, defn['class'])
            del defn["class"]
            self._pre = klass(self.gan, defn).cuda()
            self.gan.add_component("generator_pre", self._pre)

        if self.config.post:
            defn = self.config.post.copy()
            klass = GANComponent.lookup_function(None, defn['class'])
            del defn["class"]
            self._post = klass(self.gan, defn).cuda()
            self.gan.add_component("generator_post", self._post)
        print("PRE", self._pre)
        if self.config.upsample:
            self.upsample = nn.Upsample(self.config.upsample, mode='bilinear')

    def set_trainable(self, flag):
        for p in self._generator.parameters():
            if self.config.trainable == False:
                p.requires_grad = False
            else:
                p.requires_grad = flag
        for p in self._pre.parameters():
            p.requires_grad = flag
        for p in self._post.parameters():
            p.requires_grad = flag

    def parameters(self):
        return list(self._generator.parameters()) + list(self._pre.parameters()) + list(self._post.parameters())

    def teacher(self, z, G, class_idx=None, labels = None):
        device = self.gan.device
        batch_sz = self.gan.batch_size()
        if G.c_dim != 0:
            # sample random labels if no class idx is given
            if class_idx is None:
                class_indices = np.random.randint(low=0, high=G.c_dim, size=(batch_sz))
                class_indices = torch.from_numpy(class_indices).to(device)
                w_avg = G.mapping.w_avg.index_select(0, class_indices)
            else:
                w_avg = G.mapping.w_avg[class_idx].unsqueeze(0).repeat(batch_sz, 1)
                class_indices = torch.full((batch_sz,), class_idx).to(device)

            if labels is None:
                labels = F.one_hot(class_indices, G.c_dim)

        else:
            w_avg = G.mapping.w_avg.unsqueeze(0)
            if labels is not None:
                labels = None
            if class_idx is not None:
                print('Warning: --class is ignored when running an unconditional network')

        w = G.mapping(z, labels)

        w_avg = w_avg.unsqueeze(1).repeat(1, G.mapping.num_ws, 1)
        truncation_psi = 1.0
        w = w_avg + (w - w_avg) * truncation_psi
        g = G.synthesis(w, noise_mode='const')
        return g, labels

    def forward_sample(self, z):

        if self.config.mapping_only:
            return self._generator.synthesis(self._pre(z), noise_mode='const')
        return self.forward(z)

    def forward(self, z):
        if self.config.mapping_only:
            result = self._pre(z)
        elif self.config.pre:
            #w = self._generator.mapping(z, None)
            #print("W", w.shape)
            #result = self._generator.synthesis(self._pre(z), noise_mode='const')
            if self.config.skip_mapping:
                result = self._generator.synthesis(self._pre(z), noise_mode='const')
            else:
                result = self.teacher(self._pre(z), self._generator)[0]
        else:
            result = self.teacher(z, self._generator)[0]
        if self.config.post:
            result = self._post(result)
        if self.config.upsample:
            result = self.upsample(result)
        return result

    def latent_parameters(self):
        if self.config.pre:
            return [list(self._pre.parameters())[0]]
        result = list(self._generator.parameters())[1]
        for p in self._generator.parameters():
            print("SXL layer", p.shape)
        print("LATENT", result.shape)
        return [result]
