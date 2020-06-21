import torch.nn as nn

class Add(nn.Module):
    def __init__(self, layers, layer_names):
        super(Add, self).__init__()
        self.layers = layers
        self.layer_names = layer_names

    def forward(self, net, context):
        output = net
        for layer, layer_name in zip(self.layers, self.layer_names):
            if layer_name == "modulated_conv2d":
                layer_output = layer(net, context['w'])
            else:
                layer_output = layer(net)
            output = output + layer_output
        return output
