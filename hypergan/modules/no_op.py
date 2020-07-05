import torch.nn as nn

class NoOp(nn.Module):
    def __init__(self):
        super(NoOp, self).__init__()
    def forward(self, x):
        return x
