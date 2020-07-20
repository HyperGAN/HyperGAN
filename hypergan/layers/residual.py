import torch.nn as nn
import hypergan as hg

class Residual(hg.Layer):
    def __init__(self, component, args, options):
        super(Residual, self).__init__(component, args, options)
        self.size = component.current_size
        layers = []
        for i in range(options.count or 3):
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
