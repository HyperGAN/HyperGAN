import torch
import torch.nn as nn

class Attention(nn.Module):
    """ Self attention Layer from https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py """
    def __init__(self,in_dim):
        super(Attention,self).__init__()
        self.chanel_in = in_dim
        self.f = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.g = nn.Conv2d(in_channels = in_dim , out_channels = in_dim, kernel_size= 1)
        self.h = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.v = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.softmax  = nn.Softmax(dim=1) #
    def forward(self,x):
        m_batchsize,C,width ,height = x.size()
        f  = self.f(x).view(m_batchsize,C,width*height).permute(0,2,1)
        g =  self.g(x).view(m_batchsize,C,width*height)
        fg =  torch.bmm(f,g)
        attention_map = self.softmax(fg)
        h = self.h(x).view(m_batchsize,C,width*height)

        fgh = torch.bmm(h, attention_map )
        return self.v(fgh.view(x.shape))
