import torch.nn as nn
import hypergan as hg
from hypergan.layer_shape import LayerShape
from torch.functional import F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim = -1)


class Embed(hg.Layer):

    """
        ---
        description: 'layer embed for configurable component'
        ---

        # embed layer

        ## Optional arguments

            time_tokens - add sinusoidal time token embedding for sequence data
        ## input size

        Any 3-d tensor


        ## syntax

        ```json
          "reshape 768*64",
          "embed 512",
          "reshape 512*64"
        ```

    """
    def __init__(self, component, args, options):
        super(Embed, self).__init__(component, args, options)
        output_height = args[0]
        channels = component.current_size.channels
        self.size = LayerShape(channels, output_height)
        self.one_hot = False
        if options.one_hot is not None:
            self.one_hot = True
            self.num_classes = args[0]
        elif options.num_embeddings is not None:
            self.net = nn.Embedding(options.num_embeddings, output_height)
        else:
            height = component.current_size.height
            self.net = nn.Linear(height, output_height)
            nn.init.normal_(self.net.weight, 0, 1e-5)
        #self.norm = nn.LayerNorm(output_height)
        #if options.time_tokens:
        #    self.time_embedding = nn.Sequential(
        #        SinusoidalPosEmb(dim),
        #        nn.Linear(dim, time_cond_dim),
        #        nn.SiLU()
        #    )
        #else:
        self.time_embedding = None
    def forward(self, input, context):
        #if self.time_embedding is not None:
        #    self.time_embedding

        if self.one_hot:
            return F.one_hot(input, num_classes=self.num_classes)
        return self.net(input)
        #return self.norm(self.net(input))

    def output_size(self):
        return self.size
