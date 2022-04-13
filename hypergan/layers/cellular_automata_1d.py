#adapted from https://github.com/belkakari/cellular-automata-pytorch
import torch.nn as nn
import hypergan as hg
from hypergan.layer_shape import LayerShape

import torch.nn.functional as F

import torch

#Converted to conv1d:
class Perception(nn.Module):
    def __init__(self, channels=16, norm_kernel=False):
        super().__init__()
        self.channels = channels
        sobel_x = torch.tensor([-1.0, 0.0, 1.0]) / 2
        sobel_y = torch.tensor([1.0, 0.0, -1.0]) / 2
        identity = torch.tensor([0.0, 1.0, 0.0])

        self.kernel = torch.stack((identity, sobel_x, sobel_y)).repeat(channels, 1).unsqueeze(1)
        if norm_kernel:
            self.kernel /= channels

    def forward(self, state_grid):
        return F.conv1d(state_grid, 
                        self.kernel.to(state_grid.device), 
                        groups=self.channels,
                        padding=1)

class Policy(nn.Module):
    def __init__(self, state_dim=16, interm_dim=128,
                 kernel=1, padding=0,
                 bias=False):
        super().__init__()
        dim = state_dim * 3
        self.conv1 = nn.Conv1d(dim, interm_dim, kernel, padding=padding)
        self.conv2 = nn.Conv1d(interm_dim, state_dim, kernel, padding=padding,
                               bias=bias)
        nn.init.constant_(self.conv2.weight, 0.)
        if bias:
            nn.init.constant_(self.conv2.bias, 0.)

    def forward(self, state):
        interm = self.conv1(state)
        interm = torch.relu(interm)
        return self.conv2(interm)



class CellularAutomataModule(nn.Module):

    def __init__(self, channels, expansion_factor = 1):
        super().__init__()
        self.perception = Perception(channels = channels)
        self.policy = Policy(state_dim = channels, interm_dim = channels * expansion_factor)

    def alive_mask(self, state_grid):
        thr = 0.1
        alpha = state_grid[:, [1], :].clamp(0, 1)
        alive = (nn.MaxPool1d(3, stride=1, padding=1)(alpha) > thr).float()#.unsqueeze(1)
        return alive

    def forward(self, input):
        alive_pre = self.alive_mask(input)
        perception = self.perception(input)
        ds_grid = self.policy(perception)
        #skipped, stochastic mask
        input = input + ds_grid
        alive_post = self.alive_mask(input)
        final_mask = (alive_post.bool() & alive_pre.bool()).float() * input
        return final_mask


class CellularAutomata1D(hg.Layer):
    def __init__(self, component, args, options):
        super(CellularAutomata1D, self).__init__(component, args, options)
        expansion_factor = options.expansion_factor or 1
        self.size = component.current_size
        self.ca = CellularAutomataModule(self.size.dims[0], expansion_factor)

    def forward(self, input, context):
        return self.ca(input)

    def output_size(self):
        return self.size
