import torch.nn as nn
import hypergan as hg
import torch
import math
import torch.nn.functional as F
from hypergan.layer_shape import LayerShape
from hypergan.modules.multi_head_attention import  MultiHeadAttention
import hyperchamber as hc
from vit_pytorch import ViT, SimpleViT
from vit_pytorch.efficient import ViT as EfficientViT

from einops import rearrange, repeat

class ViTLayer(hg.Layer):
    """
        ---
        description: 'layer vit(vision transformer)'
        ---

        WIP

        ## syntax

        ```json
          "vit 1024"
        ```
    """
    def __init__(self, component, args, options):
        super(ViTLayer, self).__init__(component, args, options)

        self.size = LayerShape(args[0])
        heads = options.heads or 16
        depth = options.depth or 6
        mlp_dim = options.mlp_dim or 2048
        dropout = options.dropout or 0
        emb_dropout = options.emb_dropout or 0
        patch_size = options.patch_size or 32
        dim = options.dim or 1024
        self.use_transformer = options.use_transformer or False
        self.transpose = options.transpose or False
        if options.vit_type == 'simple':
            self.vit = SimpleViT(image_size = component.current_size.width, patch_size = patch_size, num_classes = args[0], dim = dim, depth = depth, heads = heads, mlp_dim = mlp_dim)
        elif options.vit_type == 'efficient':
            from x_transformers import Encoder
            efficient_transformer = Encoder(dim = dim, depth = depth, heads = heads, ff_glu = True, residual_attn = True)
            self.vit = EfficientViT(image_size = component.current_size.width, patch_size = patch_size, num_classes = args[0], transformer = efficient_transformer, dim = dim)
        else:
            self.vit = ViT( image_size = component.current_size.width,
                    patch_size = patch_size,
                    num_classes = args[0],
                    channels = component.current_size.channels,
                    dim = dim,
                    depth = depth,
                    heads = heads,
                    mlp_dim = mlp_dim,
                    dropout = dropout,
                    emb_dropout = emb_dropout)

    def output_size(self):
        return self.size

    def forward(self, input, context):

        if self.use_transformer:
            x = self.vit.to_patch_embedding(input)
            b, n, _ = x.shape

            cls_tokens = repeat(self.vit.cls_token, '1 1 d -> b 1 d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.vit.pos_embedding[:, :(n + 1)]

            vit = self.vit.transformer(x)
            if self.transpose:
                vit = vit.transpose(1, 2)
            #print('vv', vit.shape)
            return vit
        return self.vit(input)
