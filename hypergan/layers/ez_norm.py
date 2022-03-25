import torch.nn as nn
import hypergan as hg
from hypergan.layer_shape import LayerShape
from hypergan.layers.ntm import NTMLayer

class EzNorm(hg.Layer):
    """
        ---
        description: 'layer ez_norm for configurable component'
        ---

        # ez_norm layer

        `ez_norm` is a custom normalization technique that uses a conv of the input by a linear projection of a style vector.

        ## Optional arguments

            `style` - The name of the style vector to use. Defaults to "w"

        ## input size

        Any 4-d tensor

        ## output size

        Same as input size

        ## syntax

        ```json
          "ez_norm style=[style vector name]"
        ```

        ## examples

        ```json
          "latent name=w",
          ...
          "cat self (ez_norm style=w)"
        ```
    """
    def __init__(self, component, args, options):
        super(EzNorm, self).__init__(component, args, options)
        self.dim = options.dim or 1
        self.size = LayerShape(*component.current_size.dims)

        style_size = component.layer_output_sizes[options.style or 'w'].size()
        #conv_size = component.layer_output_sizes[options.conv_psm or 'r'].size()
        channels = component.current_size.channels
        dims = len(component.current_size.dims)

        options["input_size"] = component.layer_output_sizes['w'].size()
        self.beta = NTMLayer(component, [channels], options)
        self.gamma = NTMLayer(component, [channels], options)

        component.nn_init(self.beta, options.initializer)
        component.nn_init(self.gamma, options.initializer)

    def calc_mean_std(self, feat, eps=1e-5):
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def calc_mean_std1d(self, feat, eps=1e-5):
        size = feat.size()
        assert (len(size) == 3)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)
        return feat_mean, feat_std


    def forward(self, content, context, epsilon=1e-5):
        style = context['w']
        style = style.view(content.shape[0], -1)
        gamma = self.gamma(style, context)
        beta = self.beta(style, context)
        if len(content.shape) == 4:
            c_mean, c_var = self.calc_mean_std(content, epsilon)
        elif len(content.shape) == 3:
            c_mean, c_var = self.calc_mean_std1d(content, epsilon)

        c_std = (c_var + epsilon).sqrt()
        return (1+gamma.view(c_std.shape)) * ((content - c_mean) / c_std) + beta.view(c_std.shape)

    def output_size(self):
        return self.size

