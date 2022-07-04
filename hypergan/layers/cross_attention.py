#adapted from lucidrains/vit-pytorch/-/blob/main/vit_pytorch/cross_vit.py

from hypergan.layer_shape import LayerShape
from torch import nn, einsum
import hypergan as hg
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from einops_exts import rearrange_many, repeat_many, check_shape
from einops_exts.torch import EinopsToAndFrom

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# pre-layernorm

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# feedforward

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# attention

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer encoder, for small and large patches

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        print("Found", x.shape)
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        print("out", x.shape)
        x = self.project_out(x)
        return x

class CrossAttentionModule(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        norm_context = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if norm_context else nn.Identity()

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential( nn.Linear(inner_dim, dim, bias = False),
            nn.LayerNorm(dim)
        )

    def forward(self, x, context, mask = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> b h 1 d', h = self.heads,  b = b)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class EinopsToAndFromSequential(nn.Module):
    def __init__(self, from_einops, to_einops, fns):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fns = nn.ModuleList(fns)

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        for i, fn in enumerate(self.fns):
            if i == 0:
                x = fn(x, **kwargs)
            else:
                x = x + fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            groups = 8,
            norm = True
            ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else nn.Identity()
        self.activation = nn.SiLU()
        self.project = nn.Conv2d(dim, dim_out, 3, padding = 1)

    def forward(self, x, scale_shift = None):
        x = self.groupnorm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.project(x)


class CrossAttention(hg.Layer):
    """
        ---
        description: 'layer cross_attention for configurable component'
        ---

        # cross_attention layer

	Applies a CrossAttention layer between two different input sources

        ## required arguments

	    `input` - the name of the second input to combine

        ## optional arguments

            `depth` - number of layers deep
            `heads` - attention heads(default 1)
            `dim_head` - size of head(default 64)

        ## input size

        Any tensor

        ## output size

	Same as preceding layer

        ## syntax

        ```json
            "cross_attention [layer name] depth=1 heads=4"
        ```

        ## examples

    """
    def __init__(self, component, args, options):
        super(CrossAttention, self).__init__(component, args, options)
        self.size = component.current_size
        self.layer_name = args[0]
        sm_dim = self.size.channels
        lg_dim = component.layer_output_sizes[self.layer_name].height

        depth = options.depth or 1
        heads = options.heads or 8
        dim_head = options.dim_head or 64

        self.cross_attn = EinopsToAndFrom(
            'b c h w',
            'b (h w) c',
            CrossAttentionModule(
                dim = sm_dim,
                context_dim = lg_dim,
                heads = heads
            )
            ).to('cuda:0')

        self.block2 = Block(sm_dim, sm_dim, groups = 8)
        self.res_conv = nn.Conv2d(sm_dim, sm_dim, 1) if sm_dim != sm_dim else nn.Identity()

    def output_size(self):
        return self.size

    def forward(self, x, context):
        other = context[self.layer_name]
        #h = self.block1(x)
        h = x
        h = self.cross_attn(h, context = other).view(other.shape[0], *self.size.dims)# + h

        h = self.block2(h)
        return h + self.res_conv(x)

