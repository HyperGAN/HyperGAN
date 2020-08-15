import torch
from torch import nn
from torch.nn import functional as f
from hypergan.layer_shape import LayerShape
import hypergan as hg
from ..layer import Layer

class EfficientAttention(Layer):
    """
        ---
        description: 'layer efficient_attention for configurable component'
        ---

        # efficient_attention layer

        From https://github.com/cmsflash/efficient-attention

        ## input size

        Any 4-d tensor

        ## output size

        Same

        ## syntax

        ```json
          "efficient_attention"
        ```

        ## examples

        ```json
          "add self (efficient_attention)"
        ```
    """

    def __init__(self, component, args, options):
        super(EfficientAttention, self).__init__(component, args, options)
        self.dims = list(component.current_size.dims).copy()
        in_dim = self.dims[0]
        out_dim = in_dim
        if(len(args) > 0):
            out_dim = args[0]
    
        self.in_channels = in_dim
        self.key_channels = options.key_channels or 16
        self.head_count = options.heads or 4
        self.value_channels = options.value_channels or 16

        self.keys = nn.Conv2d(in_dim, self.key_channels, 1)
        self.queries = nn.Conv2d(in_dim, self.key_channels, 1)
        self.values = nn.Conv2d(in_dim, self.value_channels, 1)
        self.reprojection = nn.Conv2d(self.value_channels, out_dim, 1)

        self.size = LayerShape(out_dim, self.dims[1], self.dims[2])

    def output_size(self):
        return self.size

    def forward(self, input_, context):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = f.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            query = f.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value

        return attention.view(*([input_.shape[0]]+list(self.output_size().dims)))
