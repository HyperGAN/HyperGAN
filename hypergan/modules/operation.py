import torch.nn as nn

class Operation(nn.Module):
    def __init__(self, layers, layer_names, operation):
        super(Operation, self).__init__()
        self.layers = layers
        self.layer_names = layer_names
        self.operation = operation
        for i, (layer, layer_name) in enumerate(zip(self.layers, self.layer_names)):
            self.add_module('layer_'+str(i)+"_"+layer_name, layer)

    def forward(self, net, context):
        output = None
        for layer, layer_name in zip(self.layers, self.layer_names):
            if layer_name == "modulated_conv2d":
                layer_output = layer(net, context['w'])
            elif layer_name == "self":
                layer_output = net
            elif layer_name.split(" ")[0] == 'layer':
                layer_output = context[layer_name.split(" ")[1]]
            else:
                layer_output = layer(net)
            if output is None:
                output = layer_output
            else:
                if self.operation == "+":
                    output = output + layer_output
                elif self.operation == "*":
                    output = output * layer_output
                else:
                    raise ValidationException("Unknown operation: "+ self.operation)
        return output
