import torch
import torch.nn as nn
import numpy as np
import hypergan as hg
from hypergan.layer_shape import LayerShape
from hypergan.generators.stylegan3_generator import FullyConnectedLayer
from torch_utils import persistence

class SynthesisInputModule(nn.Module):
    def __init__(self, w_dim, channels, size, sampling_rate, bandwidth):
        super().__init__()
        self.w_dim = w_dim
        self.channels = channels
        self.size = np.broadcast_to(np.asarray(size), [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth


 
        freqs = torch.randn([self.channels, 2]).cuda()
        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= self.bandwidth
        phases = torch.rand([self.channels]) - 0.5

        self.weight = torch.nn.Parameter(torch.randn([channels, channels], device='cuda:0'))
        self.affine = FullyConnectedLayer(w_dim, 4, weight_init=0, bias_init=[1,0,0,0])
        self.register_buffer('transform', torch.eye(3, 3)) # User-specified inverse transform wrt. resulting image.
        self.register_buffer('freqs', freqs)        # [self.channels, 2]
        self.register_buffer('phases', phases)      # [self.channels]
        #self.register_parameter('weight', weight)
        self.transform.cuda()
        self.freqs.cuda()
        self.phases.cuda()

    def forward(self, w):
        # Introduce batch dimension.
        transforms = self.transform.unsqueeze(0) # [batch, row, col]
        freqs = self.freqs.unsqueeze(0) # [batch, channel, xy]
        phases = self.phases.unsqueeze(0) # [batch, channel]
        w = w.view(w.shape[0], -1)

        # Apply learned transformation.
        t = self.affine(w) # t = (r_c, r_s, t_x, t_y)
        t = t / t[:, :2].norm(dim=1, keepdim=True) # t' = (r'_c, r'_s, t'_x, t'_y)
        m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse rotation wrt. resulting image.
        m_r[:, 0, 0] = t[:, 0]  # r'_c
        m_r[:, 0, 1] = -t[:, 1] # r'_s
        m_r[:, 1, 0] = t[:, 1]  # r'_s
        m_r[:, 1, 1] = t[:, 0]  # r'_c
        m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse translation wrt. resulting image.
        m_t[:, 0, 2] = -t[:, 2] # t'_x
        m_t[:, 1, 2] = -t[:, 3] # t'_y
        transforms = m_r @ m_t @ transforms # First rotate resulting image, then translate, and finally apply user-specified transform.

        # Transform frequencies.
        phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transforms[:, :2, :2]

        # Dampen out-of-band frequencies that may occur due to the user-specified transform.
        amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)

        # Construct sampling grid.
        theta = torch.eye(2, 3, device=w.device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False)

        # Compute Fourier features.
        x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) # [batch, height, width, channel]
        x = x + phases.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (np.pi * 2))
        x = x * amplitudes.unsqueeze(1).unsqueeze(2)

        # Apply trainable mapping.
        weight = self.weight / np.sqrt(self.channels)
        x = x @ weight.t()

        # Ensure correct shape.
        x = x.permute(0, 3, 1, 2) # [batch, channel, height, width]
        return x


class SynthesisInput(hg.Layer):
    def __init__(self, component, args, options):
        super().__init__(component, args, options)
        self._output_size = component.current_size
        # Draw random frequencies from uniform 2D disc.
        self.num_layers = options.num_layers or 5
        self.num_critical = options.num_critical or 1
        self.channels = options.channels or component.current_size.channels
        w_dim = options.w_dim or component.current_size.size()
        self.img_resolution = component.gan.width()
        sampling_rate, bandwidth = self.get_sampling_rates()
        size = options.width or component.current_size.width#int(self.sampling_rate+20)

       # Setup parameters and buffers.
        self.synthesis_input = SynthesisInputModule(w_dim, self.channels, size, sampling_rate, bandwidth)

    def output_size(self):
        return self._output_size

    def forward(self, x, context={}):
        return self.synthesis_input(x)

    def get_sampling_rates(self):
        first_stopband      = 2**2.1
        last_stopband_rel   = 2**0.3
        first_cutoff = 2
        last_cutoff = self.img_resolution / 2 # f_{c,N}
        last_stopband = last_cutoff * last_stopband_rel # f_{t,N}
        exponents = np.minimum(np.arange(self.num_layers + 1) / (self.num_layers - self.num_critical), 1)
        cutoffs = first_cutoff * (last_cutoff / first_cutoff) ** exponents # f_c[i]
        stopbands = first_stopband * (last_stopband / first_stopband) ** exponents # f_t[i]

        # Compute remaining layer parameters.
        sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands * 2, self.img_resolution)))) # s[i]
        return sampling_rates[0], cutoffs[0]
 
