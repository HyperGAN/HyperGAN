import torch.nn as nn
import hypergan as hg
from hypergan.layer_shape import LayerShape

from hypergan.modules.modulated_conv2d import ModulatedConv2d as ModulatedConv2dModule
from hypergan.modules.modulated_conv2d import EqualLinear

class ModulatedConv2d(hg.Layer):
    """
        ---
        description: 'layer modulated_conv2d for configurable component'
        ---

        # modulated_conv2d layer

        Modulated conv2d from stylegan. This is slower and not recommended.

        ## syntax

        ```json
          "modulated_conv2d"
        ```

        ## examples

        ```json
          "modulated_conv2d style=w"
        ```
    """
    def __init__(self, component, args, options):
        super(ModulatedConv2d, self).__init__(component, args, options)
        channels = component.current_size.channels
        if len(args) > 0:
            channels = args[0]
        method = "conv"
        if len(args) > 1:
            method = args[1]
        upsample = method == "upsample"
        downsample = method == "downsample"

        demodulate = True
        if options.demodulate == False:
            demodulate = False

        filter = 3
        if options.filter:
            filter = options.filter

        lr_mul = 1.0
        if options.lr_mul:
            lr_mul = options.lr_mul
        input_channels = component.current_size.channels
        if options.input_channels:
            input_channels = options.input_channels

        self.layer = ModulatedConv2dModule(input_channels, channels, filter, component.layer_output_sizes['w'].size(), upsample=upsample, demodulate=demodulate, downsample=downsample, lr_mul=lr_mul)

        if upsample:
            self.size = LayerShape(channels, component.current_size.height * 2, component.current_size.width * 2)
        elif downsample:
            self.size = LayerShape(channels, component.current_size.height // 2, component.current_size.width // 2)
        else:
            raise ValidationException("Must upsample or downsample")

    def output_size(self):
        return self.size

    def forward(self, input, context):
        return self.layer.forward(input, context["w"])


