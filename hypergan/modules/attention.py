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
        self.o = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        #self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        f  = self.f(x).view(m_batchsize,width*height, -1)
        g =  self.g(x).view(m_batchsize,width*height, -1).permute(0,2,1)
        s =  torch.bmm(g,f)
        beta = self.softmax(s)
        h = self.h(x).view(m_batchsize,width*height,-1)

        out = torch.bmm(beta, h.permute(0,2,1) )
        out = self.o(out.view(x.shape))
        out = out.view(m_batchsize,C,width,height)
        
        return out
