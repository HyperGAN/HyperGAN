import torch.nn as nn
import hypergan as hg
import torch
import math
import torch.nn.functional as F
from hypergan.layer_shape import LayerShape
from hypergan.modules.multi_head_attention import  MultiHeadAttention
import hyperchamber as hc
import torch
from x_unet import XUnet

class XUnetLayer(hg.Layer):
    """
        ---
        description: 'layer xunet'
        ---

        WIP

        ## syntax

        ```json
          "xunet 1024"
        ```
    """
    def __init__(self, component, args, options):
        super(XUnetLayer, self).__init__(component, args, options)

        self.size = component.current_size
        self.unet = XUnet(
                    dim = 64,
                    dim_mults = (1, 2, 4),
                    num_blocks_per_stage = (2, 2, 2),
                    num_self_attn_per_stage = (0, 0, 1),
                    nested_unet_depths = (4, 2, 1),     # nested unet depths, from unet-squared paper
                    consolidate_upsample_fmaps = True,     # whether to consolidate outputs from all upsample blocks, used in unet-squared paper
                )

    def output_size(self):
        return self.size

    def forward(self, input, context):
        return self.unet(input)
