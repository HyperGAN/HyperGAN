import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, output_size, heads):
        super(MultiHeadAttention,self).__init__()
        self.heads = heads
        self.features = input_size // heads
        self.features_sqrt = math.sqrt(self.features)

        self.f = nn.Linear(input_size, input_size)
        self.g = nn.Linear(input_size, input_size)
        self.h = nn.Linear(input_size, input_size)
        self.o = nn.Linear(input_size, output_size)

        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        batch_size, input_size = x.shape
        f  = self.f(x).view(batch_size,1,self.heads,self.features).permute(0,2,3,1)
        g =  self.g(x).view(batch_size,1,self.heads,self.features).permute(0,2,1,3)
        fg =  torch.matmul(g, f) / self.features_sqrt
        attention_map = self.softmax(fg)
        h = self.h(x).view(batch_size,1,self.heads,self.features).permute(0,2,1,3)

        fgh = torch.matmul(attention_map, h)
        output = fgh.permute(0,2,1,3).view(x.shape)
        return self.o(output)
