import torch.nn as nn

class Layer(nn.Module):
    def __init__(self, component, args, options):
        super(Layer, self).__init__()
        self.args = args
        self.options = options

    def forward(self, input, context):
        pass

    def output_size(self):
        pass

    def latent_parameters(self):
        return []
