from .base_generator import BaseGenerator
from hypergan.generators import legacy
import numpy as np
import torch

class StyleganXLGenerator(BaseGenerator):
    def create(self):
        with open(self.config.pkl, "rb") as f:
            G = legacy.load_network_pkl(f)['G_ema']
            self._generator = G.eval().requires_grad_(True).to(self.gan.device)

    def set_trainable(self, flag):
        for p in self._generator.parameters():
            p.requires_grad = flag

    def parameters(self):
        return self._generator.parameters()

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

    def forward(self, z):
        return self.teacher(z, self._generator)[0]
