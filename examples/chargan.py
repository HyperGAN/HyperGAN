from common import *
from typing import Union, Callable, Optional
from hypergan.viewer import GlobalViewer
from hyperchamber import Config
from hypergan.discriminators import *
from hypergan.distributions import *
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.gans.base_gan import BaseGAN
from hypergan.generators import *
from hypergan.inputs import *
from hypergan.layer_shape import LayerShape
from hypergan.samplers import *
from hypergan.trainers import *
from hypergan.discriminators.base_discriminator import BaseDiscriminator
from torch import nn
import torch.nn.functional as F
import random
import argparse
import copy
import hyperchamber as hc
import hypergan as hg
import importlib
import json
import numpy as np
import os
import re
import string
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import uuid
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def _split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results

class NTM(nn.Module):
    """A Neural Turing Machine."""
    def __init__(self, num_inputs, num_outputs, controller, memory, heads):
        """Initialize the NTM.

        :param num_inputs: External input size.
        :param num_outputs: External output size.
        :param controller: :class:`LSTMController`
        :param memory: :class:`NTMMemory`
        :param heads: list of :class:`NTMReadHead` or :class:`NTMWriteHead`

        Note: This design allows the flexibility of using any number of read and
              write heads independently, also, the order by which the heads are
              called in controlled by the user (order in list)
        """
        super(NTM, self).__init__()

        # Save arguments
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller = controller
        self.memory = memory
        self.heads = heads

        self.N, self.M = memory.size()
        _, self.controller_size = controller.size()

        # Initialize the initial previous read values to random biases
        self.num_read_heads = 0
        self.init_r = []
        for head in heads:
            if head.is_read_head():
                init_r_bias = torch.randn(1, self.M) * 0.01
                self.register_buffer("read{}_bias".format(self.num_read_heads), init_r_bias.data)
                self.init_r += [init_r_bias]
                self.num_read_heads += 1

        assert self.num_read_heads > 0, "heads list must contain at least a single read head"

        # Initialize a fully connected layer to produce the actual output:
        #   [controller_output; previous_reads ] -> output
        self.fc = nn.Linear(self.controller_size + self.num_read_heads * self.M, num_outputs)
        self.reset_parameters()

    def create_new_state(self, batch_size):
        init_r = [r.clone().repeat(batch_size, 1) for r in self.init_r]
        controller_state = self.controller.create_new_state(batch_size)
        heads_state = [head.create_new_state(batch_size) for head in self.heads]

        return init_r, controller_state, heads_state

    def reset_parameters(self):
        # Initialize the linear layer
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.normal_(self.fc.bias, std=0.01)

    def forward(self, x, prev_state):
        """NTM forward function.

        :param x: input vector (batch_size x num_inputs)
        :param prev_state: The previous state of the NTM
        """
        # Unpack the previous state
        prev_reads, prev_controller_state, prev_heads_states = prev_state

        # Use the controller to get an embeddings
        inp = torch.cat([x] + prev_reads, dim=1)
        controller_outp, controller_state = self.controller(inp, prev_controller_state)

        # Read/Write from the list of heads
        reads = []
        heads_states = []
        for head, prev_head_state in zip(self.heads, prev_heads_states):
            if head.is_read_head():
                r, head_state = head(controller_outp, prev_head_state)
                reads += [r]
            else:
                head_state = head(controller_outp, prev_head_state)
            heads_states += [head_state]

        # Generate Output
        inp2 = torch.cat([controller_outp] + reads, dim=1)
        o = F.sigmoid(self.fc(inp2))

        # Pack the current state
        state = (reads, controller_state, heads_states)

        return o, state
class NTMHeadBase(nn.Module):
    """An NTM Read/Write Head."""

    def __init__(self, memory, controller_size):
        """Initilize the read/write head.

        :param memory: The :class:`NTMMemory` to be addressed by the head.
        :param controller_size: The size of the internal representation.
        """
        super(NTMHeadBase, self).__init__()

        self.memory = memory
        self.N, self.M = memory.size()
        self.controller_size = controller_size

    def create_new_state(self, batch_size):
        raise NotImplementedError

    def register_parameters(self):
        raise NotImplementedError

    def is_read_head(self):
        return NotImplementedError

    def _address_memory(self, k, β, g, s, γ, w_prev):
        # Handle Activations
        k = k.clone()
        β = F.softplus(β)
        g = F.sigmoid(g)
        s = F.softmax(s, dim=1)
        γ = 1 + F.softplus(γ)

        w = self.memory.address(k, β, g, s, γ, w_prev)

        return w


class NTMReadHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(NTMReadHead, self).__init__(memory, controller_size)

        # Corresponding to k, β, g, s, γ sizes from the paper
        self.read_lengths = [self.M, 1, 1, 3, 1]
        self.fc_read = nn.Linear(controller_size, sum(self.read_lengths))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        # The state holds the previous time step address weightings
        return torch.zeros(batch_size, self.N)

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
        nn.init.normal_(self.fc_read.bias, std=0.01)

    def is_read_head(self):
        return True

    def forward(self, embeddings, w_prev):
        """NTMReadHead forward function.

        :param embeddings: input representation of the controller.
        :param w_prev: previous step state
        """
        o = self.fc_read(embeddings)
        k, β, g, s, γ = _split_cols(o, self.read_lengths)

        # Read from memory
        w = self._address_memory(k, β, g, s, γ, w_prev)
        r = self.memory.read(w)

        return r, w


class NTMWriteHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(NTMWriteHead, self).__init__(memory, controller_size)

        # Corresponding to k, β, g, s, γ, e, a sizes from the paper
        self.write_lengths = [self.M, 1, 1, 3, 1, self.M, self.M]
        self.fc_write = nn.Linear(controller_size, sum(self.write_lengths))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        return torch.zeros(batch_size, self.N)

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.fc_write.weight, gain=1.4)
        nn.init.normal_(self.fc_write.bias, std=0.01)

    def is_read_head(self):
        return False

    def forward(self, embeddings, w_prev):
        """NTMWriteHead forward function.

        :param embeddings: input representation of the controller.
        :param w_prev: previous step state
        """
        o = self.fc_write(embeddings)
        k, β, g, s, γ, e, a = _split_cols(o, self.write_lengths)

        # e should be in [0, 1]
        e = F.sigmoid(e)

        # Write to memory
        w = self._address_memory(k, β, g, s, γ, w_prev)
        self.memory.write(w, e, a)

        return w
class FastMemory(nn.Module):
    """Memory bank for NTM."""
    def __init__(self, N, M):
        """Initialize the NTM Memory matrix.

        The memory's dimensions are (batch_size x N x M).
        Each batch has it's own memory matrix.

        :param N: Number of rows in the memory.
        :param M: Number of columns/features in the memory.
        """
        super(FastMemory, self).__init__()

        self.N = N
        self.M = M

        # The memory bias allows the heads to learn how to initially address
        # memory locations by content
        self.register_buffer('mem_bias', torch.Tensor(N, M))

        # Initialize memory bias
        stdev = 1 / (np.sqrt(N + M))
        nn.init.uniform_(self.mem_bias, -stdev, stdev)

    def reset(self, batch_size):
        """Initialize memory from bias, for start-of-sequence."""
        self.batch_size = batch_size
        self.memory = self.mem_bias.clone().repeat(batch_size, 1, 1)

    def size(self):
        return self.N, self.M

    def read(self, w):
        """Read from memory (according to section 3.1)."""
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    def write(self, w, e, a):
        """write to memory (according to section 3.2)."""
        self.prev_mem = self.memory
        self.memory = torch.Tensor(self.batch_size, self.N, self.M)
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.prev_mem * (1 - erase) + add

    def address(self, k, β, g, s, γ, w_prev):
        """NTM Addressing (according to section 3.3).

        Returns a softmax weighting over the rows of the memory matrix.

        :param k: The key vector.
        :param β: The key strength (focus).
        :param g: Scalar interpolation gate (with previous weighting).
        :param s: Shift weighting.
        :param γ: Sharpen weighting scalar.
        :param w_prev: The weighting produced in the previous time step.
        """
        # Content focus
        wc = self._similarity(k, β)

        # Location focus
        wg = self._interpolate(w_prev, wc, g)
        ŵ = self._shift(wg, s)
        w = self._sharpen(ŵ, γ)

        return w

    def _similarity(self, k, β):
        k = k.view(self.batch_size, 1, -1)
        w = F.softmax(β * F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1), dim=1)
        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, w, s):

        batch_size = w.size(0)
        max_shift = int((s.size(1) - 1) / 2)
        
        unrolled = torch.cat([w[:, -max_shift:], w, w[:, :max_shift]], 1)
        return F.conv1d(unrolled.unsqueeze(1), s.unsqueeze(1))[range(batch_size), range(batch_size)]
        result = torch.zeros(wg.size(), device=s.device)
        for b in range(self.batch_size):
            result[b] = _convolve(wg[b], s[b])
        return result

    def _sharpen(self, ŵ, γ):
        w = ŵ ** γ
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w
class SimpleController(nn.Module):
    """An NTM controller based on LSTM."""
    def __init__(self, num_inputs, num_outputs, num_layers):
        super(SimpleController, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers

        #self.lstm = nn.LSTM(input_size=num_inputs,
        #                    hidden_size=num_outputs,
        #                    num_layers=num_layers)
        self.net = nn.Sequential(
            nn.Linear(num_inputs, num_outputs),
            nn.Hardtanh()
            #nn.Conv1d(inner_dim, inner_dim*2, kernel_size = 1),
            #nn.PReLU()
        ).cuda()

        # The hidden state is a learned parameter
        #self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)
        #self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)

        self.reset_parameters()

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        #lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        #lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        return None, None#lstm_h, lstm_c

    def reset_parameters(self):
        for p in self.net.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs +  self.num_outputs))
                nn.init.uniform_(p, -stdev, stdev)

    def size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, x, prev_state):
        outp = self.net(x)
        return outp.squeeze(0), prev_state

class NTMInput:
    def __init__(self, batch_size):
        self._batch_size = batch_size

    def next(self, index=0):
        return self.ones([self._batch_size, 1]).cuda()*3.0

    def batch_size(self):
        return self._batch_size


class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm_first = norm_first

        #self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        #self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        x = self.activation(x)
        x = x + self.activation(self._sa_block(x, src_mask, src_key_padding_mask))
        x = x + self.activation(self._ff_block(x))
        #if self.norm_first:
        #    x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        #    x = x + self._ff_block(self.norm2(x))
        #else:
        #    x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        #    x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return x

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.activation(self.linear1(x)))
        return x

#class SynthesisInputModule1d(nn.Module):
#    def __init__(self, w_dim, channels, size, sampling_rate, bandwidth):
#        super().__init__()
#        self.w_dim = w_dim
#        self.channels = channels
#        self.size = np.broadcast_to(np.asarray(size), [1])
#        self.sampling_rate = sampling_rate
#        self.bandwidth = bandwidth
#        freqs = torch.randn([self.channels, 1]).cuda()
#        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
#        freqs /= radii * radii.square().exp().pow(0.25)
#        freqs *= self.bandwidth
#        phases = torch.rand([self.channels]) - 0.5
#
#        self.weight = torch.nn.Parameter(torch.randn([channels, channels], device='cuda:0'))
#        self.affine = FullyConnectedLayer(w_dim, 2, weight_init=0, bias_init=[1,0])
#        self.register_buffer('transform', torch.eye(2, 2)) # User-specified inverse transform wrt. resulting image.
#        self.register_buffer('freqs', freqs)        # [self.channels, 2]
#        self.register_buffer('phases', phases)      # [self.channels]
#        #self.register_parameter('weight', weight)
#        self.transform.cuda()
#        self.freqs.cuda()
#        self.phases.cuda()
#
#    def forward(self, w):
#        # Introduce batch dimension.
#        transforms = self.transform.unsqueeze(0) # [batch, row, col]
#        freqs = self.freqs.unsqueeze(0) # [batch, channel, xy]
#        phases = self.phases.unsqueeze(0) # [batch, channel]
#        w = w.view(w.shape[0], -1)
#
#        # Apply learned transformation.
#        t = self.affine(w) # t = (r_c, r_s, t_x, t_y)
#        t = t / t[:, :1].norm(dim=1, keepdim=True) # t' = (r'_c, r'_s, t'_x, t'_y)
#        m_r = torch.eye(2, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse rotation wrt. resulting image.
#        m_r[:, 0, 0] = t[:, 0]  # r'_c
#        m_r[:, 0, 1] = -t[:, 1] # r'_s
#        m_r[:, 1, 0] = t[:, 1]  # r'_s
#        m_r[:, 1, 1] = t[:, 0]  # r'_c
#        m_t = torch.eye(2, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse translation wrt. resulting image.
#        m_t[:, 0, 1] = -t[:, 1] # t'_x
#        transforms = m_r @ m_t @ transforms # First rotate resulting image, then translate, and finally apply user-specified transform.
#
#        # Transform frequencies.
#        phases = phases + (freqs @ transforms[:, :1, 1:]).squeeze(2)
#        freqs = freqs @ transforms[:, :1, :1]
#
#        # Dampen out-of-band frequencies that may occur due to the user-specified transform.
#        amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)
#
#        # Construct sampling grid.
#        theta = torch.eye(1, 2, device=w.device)
#        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
#        grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, self.size[0]], align_corners=False)
#
#        # Compute Fourier features.
#        x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) # [batch, height, width, channel]
#        x = x + phases.unsqueeze(1).unsqueeze(2)
#        x = torch.sin(x * (np.pi * 2))
#        x = x * amplitudes.unsqueeze(1).unsqueeze(2)
#
#        # Apply trainable mapping.
#        weight = self.weight / np.sqrt(self.channels)
#        x = x @ weight.t()
#
#        # Ensure correct shape.
#        x = x.permute(0, 3, 1, 2) # [batch, channel, height, width]
#        return x


class NTMGenerator(BaseGenerator):
    def create(self):
        self.output_dim = 8*1024
        dim = 768
        inner_dim = 1024
        N = 1024
        M = 32
        memory = FastMemory(N, M).cuda()
        clayers = 1
        num_heads = 2
        if self.gan.config.lstm:
            controller = LSTMController(dim + M*num_heads, inner_dim, clayers).cuda()
        else:
            controller = SimpleController(dim + M*num_heads, inner_dim, clayers).cuda()
        heads = []
        for i in range(num_heads):
            heads += [
                NTMReadHead(memory, inner_dim).cuda(),
                NTMWriteHead(memory, inner_dim).cuda()
            ]

        self.ntm = NTM(dim, self.output_dim, controller, memory, heads).cuda()

        #self.decoder = nn.Sequential(nn.ConvTranspose1d(1024, 512, stride=2, kernel_size = 4, padding=1), nn.ReLU(), 
        #        nn.ConvTranspose1d(512, 256, stride=2, kernel_size = 4, padding=1), nn.ReLU(),
        #        nn.ConvTranspose1d(256, 128, stride=2, kernel_size = 4, padding=1), nn.ReLU(),
        #        nn.ConvTranspose1d(128, 64, stride=2, kernel_size = 4, padding=1), nn.ReLU(),
        #        nn.ConvTranspose1d(64, 1, stride=2, kernel_size = 4, padding=1)
        #        ).cuda()
        #encoder_layer = TransformerEncoderLayer(d_model=self.output_dim, nhead=4)
        #self.net = torch.nn.TransformerEncoder(encoder_layer, num_layers=2).cuda()
        #self.decoder = nn.Sequential(nn.Flatten(),nn.ReLU(), nn.Linear(self.output_dim, 256)).cuda()
        self.decoder = nn.Sequential(nn.ReLU(), nn.Conv1d(self.output_dim//256, 1, kernel_size=1, stride=1, padding=0)).cuda()
        self.memory = memory
        self.controller = controller
        self.softmax = torch.nn.Softmax(dim=1)
        self.relu = torch.nn.ReLU()
        cas = []
        convs = []
        num_ca_layers = 2
        for i in range(num_ca_layers):
            convs.append(nn.Conv1d(self.output_dim//256,self.output_dim//256, stride=1, kernel_size=1, padding=0).cuda())
            cas.append(hg.layers.cellular_automata_1d.CellularAutomataModule(self.output_dim//256, 2).cuda())
        self.ca_layers = [cas, convs]

    def init_sequence(self):
        self.memory.reset(self.gan.batch_size())
        self.prev_state = self.ntm.create_new_state(self.gan.batch_size())
        init_r = [i.cuda() for i in self.prev_state[0]]
        controller_state = self.prev_state[1]
        heads_state = [c.cuda() for c in self.prev_state[2]]
        self.prev_state = (init_r, controller_state, heads_state)

    def forward(self, input, context={}):
        net = input
        net, self.prev_state = self.ntm(net, self.prev_state)
        net = net.view(net.shape[0], self.output_dim//256, 256)
        for (ca, conv) in zip(*self.ca_layers):
            net = net + conv(net) + ca(net)
            net = self.relu(net)
        #net = self.net(net.unsqueeze(1))
        net = self.decoder(net)
        net = net.view(net.shape[0], 256)
        return self.softmax(net), self.prev_state



class TextData(data.Dataset):
    def __init__(self, path, length, device, mode, one_hot):
        self.size = os.path.getsize(path)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.file = None
        self.path = path
        self.device = device
        self.length = length
        self.mode = mode
        self.one_hot = one_hot

    def encode_line(self, line):
        if self.one_hot:
            one_hot_line = [np.eye(256)[min(ord(c), 255)] for c in line]
            return torch.tensor(one_hot_line, dtype=torch.float32)
        return torch.tensor([(c/256.0 * 2 - 1) for c in line])

    def __getitem__(self, index):
        if self.file is None:
            self.file = open(self.path, 'r', encoding='ascii')
        self.file.seek(index)
        data = self.file.read(self.length)
        if self.mode == "constant":
            encoded = torch.ones_like(self.encode_line(data).view(1, -1)) * 0.5
        else:
            encoded = self.encode_line(data)
        return [encoded]

    def __len__(self):
        return self.size - self.length

    def sample_output(self, val):
        chars = []
        if self.one_hot:
            actions = [chr(i) for i in range(255)]
            actions.append("<|unk|>")
            for p in np.split(val, val.shape[0], axis=0):
                p = p.squeeze()
                chars.append(np.random.choice(actions, p=p))
            return "".join(chars)

        val = (np.reshape(val, [-1]) + 1.0) * 127.5
        x = val[0]
        val = np.round(val)
        ox_val = [chr(int(obj)) for obj in list(val)]
        string = "".join(ox_val)
        return string

class TextInput:
    def __init__(self, config, batch_size, filename, length, mode='seek', one_hot=False, device=0):
        self.textdata = TextData(filename, length, device=device, mode=mode, one_hot=one_hot)
        self.dataloader = data.DataLoader(self.textdata, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.dataset = None
        self._batch_size = batch_size
        self.length = length
        self.filename = filename
        self.one_hot = one_hot
        self.device = device
        self.config = config

    def text_plot(self, size, filename, data, x):
        bs = x.shape[0]
        data = np.reshape(data, [bs, -1])
        x = np.reshape(x, [bs, -1])
        plt.clf()
        plt.figure(figsize=(2,2))
        data = np.squeeze(data)
        plt.plot(x)
        plt.plot(data)
        plt.xlim([0, size])
        plt.ylim([-2, 2.])
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        plt.savefig(filename)

    def to(self, device):
        return TextInput(self.config, self._batch_size, self.filename, self.length, self.one_hot, device=device)

    def next(self, index=0):
        if self.dataset is None:
            self.dataset = iter(self.dataloader)
        try:
            self.sample = self.dataset.next()[0].to(self.device)
            return self.sample
        except StopIteration:
            self.dataset = iter(self.dataloader)
            return self.next(index)
        except TypeError as e:
            print("Type Error from input! Continuing")
            print(e)
            return self.next(index)

    def batch_size(self):
        return self._batch_size

    def channels(self):
        return 1#self.length

    def width(self):
        return 1

    def height(self):
        return self.length

class CharGAN(BaseGAN):
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)
        #self.x, self.chars = self.inputs.next()
        self.x = self.inputs.next()

    def build(self):
        torch.onnx.export(self.generator, torch.randn(*self.latent.z.shape, device='cuda'), "generator.onnx", verbose=True, input_names=["latent"], output_names=["generator"])

    def required(self):
        return "generator".split()

    def create(self):
        self.latent = self.create_component("latent")
        self.generator = NTMGenerator(self, self.config)
        self.add_component('generator', self.generator)
        self.discriminator = self.create_component("discriminator")

    def forward_discriminator(self, *inputs):
        return self.discriminator(inputs[0])

    def split_qa(self, x):
        split = torch.split(x, x.shape[1]//2, dim=1)
        return split[0], split[1]

    def forward_pass(self):
        #self.x, self.chars = self.inputs.next()
        self.x = self.inputs.next()
        self.q, self.a = self.split_qa(self.x)
        self.augmented_latent = self.train_hooks.augment_latent(self.latent.next())
        d_reals, d_fakes = [], []
        min_pass_score = torch.ones([self.batch_size(), 1], device=self.device, dtype=torch.float32)*0.1
        pass_score = min_pass_score.clone() #TODO
        #for c in self.q:
        #    d_real = self.forward_discriminator(c, min_pass_score)
        #    d_reals.append(d_real)
        #for c in self.a:
        #    d_real = self.forward_discriminator(a, min_pass_score)
        #    d_reals.append(d_real)
        #self.discriminator.init_sequence()
        self.generator.init_sequence()
        gc = None
        gs = []
        for c in torch.split(self.q, 1, dim=1):
            z = self.latent.next()
            #d_fake = self.forward_discriminator(c, pass_score)
            c = c.squeeze()
            zc = torch.cat([z,c], dim=1)
            gc, _ = self.generator(zc)
            #gs.append(c.unsqueeze(1))
            #d_fakes.append(d_fake)
        for c in torch.split(self.a, 1, dim=1):
            z = self.latent.next()
            gs.append(gc.unsqueeze(1))
            #d_fake = self.forward_discriminator(gc, pass_score)
            #d_fakes.append(d_fake)
            zc = torch.cat([z,gc], dim=1)
            gc, _ = self.generator(zc)
        gcat = torch.cat(gs, dim=1)
        #ce = gcat * torch.log(self.x + 1e-8)  + (1-gcat)*torch.log(1-self.x + 1e-8)
        #self.xargs = [torch.zeros_like(ce)]#torch.cat([torch.zeros_like(ce), min_pass_score], dim=1)
        #self.gargs = [ce]#torch.cat([ce, pass_score], dim=1)
        x1 = self.q
        x2 = self.a
        g1 = self.q
        g2 = gcat
        if self.config.g_only:
            self.xargs = [x2]
            self.gargs = [g2]
        else:
            self.xargs = [torch.cat([x1,x2], dim=1)]
            self.gargs = [torch.cat([g1,g2], dim=1)]

        #self.add_metric('ce', ce.mean())
        d_fake = self.forward_discriminator(*self.gargs)
        d_real = self.forward_discriminator(*self.xargs)
        self.d_fake = d_fake#sum(d_fakes)/len(d_fakes)
        self.d_real = d_real#sum(d_reals)/len(d_reals)
        return d_real, d_fake

    def forward_loss(self, loss=None):
        d_real, d_fake = self.forward_pass()
        d_loss, g_loss = loss.forward(d_real, d_fake)

        #reconstruct_chars = self.to_chars(self.x)
        #loss = ((self.chars - reconstruct_chars)**2).mean()*100
        #self.add_metric('char_loss', loss)
        return [d_loss, g_loss]# + loss]

    def input_nodes(self):
        "used in hypergan build"
        return [
        ]

    def output_nodes(self):
        "used in hypergan build"
        return [
        ]

    def discriminator_components(self):
        return [self.discriminator]

    def generator_components(self):
        return [self.generator]#, self.to_chars]

    def discriminator_fake_inputs(self):
        return [self.gargs]

    def discriminator_real_inputs(self):
        return self.xargs

if __name__ == '__main__':
    arg_parser = ArgumentParser("Learn from a text file", require_directory=False)
    arg_parser.parser.add_argument('--filename', type=str, default='chargan.txt', help='Input dataset');
    args = arg_parser.parse_args()

    def lookup_config(args):
        if args.action != 'search':
            return hg.configuration.Configuration.load(args.config+".json")
        else:
            return hc.Config({"length": 1024})

    config = lookup_config(args)

    config_name = args.config
    save_file = "saves/"+config_name+"/model.ckpt"

    mode = "seek"

    if args.action == 'search':
        mode = 'constant'

    inputs = TextInput(config, args.batch_size, args.filename, config.length, mode=mode, one_hot=config.length)

    def parse_size(size):
        width = int(size.split("x")[0])
        height = int(size.split("x")[1])
        channels = int(size.split("x")[2])
        return [width, height, channels]

    def random_config_from_list(config_list_file):
        """ Chooses a random configuration from a list of configs (separated by newline) """
        lines = tuple(open(config_list_file, 'r'))
        config_file = random.choice(lines).strip()
        print("[hypergan] config file chosen from list ", config_list_file, '  file:', config_file)
        return hg.configuration.Configuration.load(config_file+".json")

    def setup_gan(config, inputs, args):
        if "encode" in config:
            print("CHARGAN")
            gan = CharGAN(config, inputs=inputs)
        else:
            gan = hg.GAN(config, inputs=inputs)
        gan.load(save_file)

        return gan

    def sample(config, inputs, args):
        gan = setup_gan(config, inputs, args)

    def search(config, inputs, args):
        metrics = train(config, inputs, args)
        return metrics

    def train(config, inputs, args):
        gan = setup_gan(config, inputs, args)
        trainable_gan = hg.TrainableGAN(gan, save_file = save_file, devices = args.devices, backend_name = args.backend)

        trainers = []

        x_0 = gan.inputs.next()
        z_0 = gan.latent.sample()

        ax_sum = 0
        ag_sum = 0
        diversity = 0.00001
        dlog = 0
        last_i = 0
        steps = 0
        metric = 100

        def run_g(q, a):
            gs = []
            for c in torch.split(q, 1, dim=1):
                latent = gan.latent.next()
                zc = torch.cat([latent, c.squeeze()], dim=1)
                gc, _ = gan.generator(zc)
                #gs.append(c)

            for i in range(16):
                for c in torch.split(a, 1, dim=1):
                    gs.append(gc.unsqueeze(1))
                    latent = gan.latent.next()
                    zc = torch.cat([latent, gc.squeeze()], dim=1)
                    gc, _ = gan.generator(zc)
            g = torch.cat(gs, dim=1)
            g_output = inputs.textdata.sample_output(g[0].cpu().detach().numpy())
            g_output = re.sub(r'[\x00-\x09\x7f-\x9f]', '�', g_output)
            return g_output

        latent = gan.latent.sample()
        while(True):
            steps +=1
            if steps > args.steps and args.steps != -1:
                break
            trainable_gan.step()

            if args.action == 'train' and steps % args.save_every == 0 and steps > 0:
                print("saving " + save_file)
                trainable_gan.save()

            if steps % args.sample_every == 0 or steps == args.steps -1:
                q, a = gan.split_qa(x_0)
                gan.generator.init_sequence()
                gc = None

                print("Query:")
                print(inputs.textdata.sample_output(q[0][:128].cpu().numpy()))

                #req_bytes = gan.to_chars(x_val)
                #print(req_bytes)
                #chars = inputs.textdata.sample_output_chars(req_bytes[0].cpu().detach().numpy())
                #print("Q reencoded")
                #print(inputs.textdata.sample_output(req[0][:128]))
                #print("Q bytes out")
                #chars = re.sub(r'[\x00-\x10\x7f-\x9f]', '�', chars)
                #print(chars)
                #print("X answer:")
                #print(inputs.textdata.sample_output(a[0]))
                print("G answer:")
                g_output = run_g(q, a)
                print(g_output)
                gan.generator.init_sequence()
                print("G answer 2:")
                g_output = run_g(q, a)
                print(g_output)
                #print("G encoded:")
                #g_bytes = gan.to_chars(g)
                #g_output = inputs.textdata.sample_output_chars(req_bytes[0].cpu().detach().numpy())
                #g_output = re.sub(r'[\x00-\x09\x7f-\x9f]', '�', g_output)
                #print(g_output)
                print("Q mean:")
                print(q.mean())
                print("A mean:")
                print(a.mean())
                #metric = (a-g).mean().cpu().detach().numpy()
                #print("METRIC")
                #print(metric)
        return [0]#metric]

        #if args.config is None:
        #    with open("sequence-results-10k.csv", "a") as myfile:
        #        myfile.write(config_name+","+str(ax_sum)+","+str(ag_sum)+","+ str(ax_sum+ag_sum)+","+str(ax_sum*ag_sum)+","+str(dlog)+","+str(diversity)+","+str(ax_sum*ag_sum*(1/diversity))+","+str(last_i)+"\n")

    if args.action == 'train':
        metrics = train(config, inputs, args)
        print("Resulting metrics:", metrics)
    elif args.action == 'sample':
        sample(config, inputs, args)
    elif args.action == 'search':
        config_filename = "chargan-search-"+str(uuid.uuid4())+'.json'
        config = hc.Config(json.loads(open(os.getcwd()+'/chargan-search.json', 'r').read()))
        config.trainer["hooks"].append(
          {
            "class": "function:hypergan.train_hooks.inverse_train_hook.InverseTrainHook",
            "gamma": [random.choice([1, 10, 1e-1, 100]), random.choice([1, 10, 1e-1, 100])],
            "g_gamma": random.choice([1, 10, 1e-1, 100]),
            "only_real": random.choice([False, False, True]),
            "only_fake": random.choice([False, False, True]),
            "invert": random.choice([False, False, True])
          })
        config.trainer["optimizer"] = random.choice([{
            "class": "class:hypergan.optimizers.adamirror.Adamirror",
            "lr": random.choice(list(np.linspace(0.0001, 0.002, num=1000))),
            "betas":[random.choice([0.1, 0.9, 0.9074537537537538, 0.99, 0.999]),random.choice([0,0.9,0.997])]
        },{
            "class": "class:torch.optim.SGD",
            "lr": random.choice([1e-1, 1, 1e-2, 4e-2])
        },{
            "class": "class:torch.optim.RMSprop",
            "lr": random.choice([1e-3, 1e-4, 5e-4, 3e-3]),
            "alpha": random.choice([0.9, 0.99, 0.999]),
            "eps": random.choice([1e-8, 1e-13]),
            "weight_decay": random.choice([0, 1e-2]),
            "momentum": random.choice([0, 0.1, 0.9]),
            "centered": random.choice([False, True])
        },
        {

            "class": "class:torch.optim.Adam",
            "lr": 1e-3,
            "betas":[random.choice([0.1, 0.9, 0.9074537537537538, 0.99, 0.999]),random.choice([0,0.9,0.997])],
            "eps": random.choice([1e-8, 1e-13]),
            "weight_decay": random.choice([0, 1e-2]),
            "amsgrad": random.choice([False, True])
            }

        ])

     


        metric_sum = search(config, inputs, args)
        search_output = "chargan-search-results.csv"
        hc.Selector().save(config_filename, config)
        with open(search_output, "a") as myfile:
            total = sum(metric_sum)
            myfile.write(config_filename+","+",".join([str(x) for x in metric_sum])+","+str(total)+"\n")

    else:
        print("Unknown action: "+args.action)

GlobalViewer.close()
