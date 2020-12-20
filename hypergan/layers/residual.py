import torch.nn as nn
import hypergan as hg

class Residual(hg.Layer):
    """
        ---
        description: 'layer residual for configurable component'
        ---

        # residual layer

        `residual` adds one or more residual blocks https://paperswithcode.com/method/residual-block

        ## optional arguments

        The number of residual blocks to add

        ## input size

        Any 4-d tensor

        ## output size

        Same as input size

        ## syntax

        ```json
          "residual COUNT"
        ```

        ## examples

        ```json
          "residual 3"
        ```
    """


    def __init__(self, component, args, options):
        super(Residual, self).__init__(component, args, options)
        self.size = component.current_size
        layers = []
        for i in range(args[0] or 3):
            layers += [nn.Conv2d(self.size.channels, self.size.channels, 3, 1, padding = (1, 1))]
            layers += [nn.ReLU()]
            layers += [nn.Conv2d(self.size.channels, self.size.channels, 3, 1, padding = (1, 1))]
            layers += [nn.ReLU()]

        self.residual = nn.Sequential(*layers)

    def output_size(self):
        return self.size

    def forward(self, input, context):
        residual = self.residual(input)
        return input + residual
