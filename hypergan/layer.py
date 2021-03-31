import torch.nn as nn
import hypergan as hg
from hypergan.modules.no_op import NoOp

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

    def norm_layers(self, component):
        if self.options.norm == "adaptive_instance_norm":
            evonorm = component.parse_layer("evo_norm")[1]
            return [component.parse_layer("adaptive_instance_norm style=" + self.style + "")[1], evonorm]
        elif self.options.norm == "ez_norm":
            evonorm = component.parse_layer("evo_norm")[1]
            return [component.parse_layer("ez_norm style=" + self.style + "")[1], evonorm]
        elif self.options.norm == "evo_norm":
            evonorm = component.parse_layer("evo_norm")[1]
            return [evonorm]
        else:
            return []

    def forward_module_list(self, input, layer_names, layers, context):
        for name, module in zip(layer_names, layers):
            if isinstance(module, hg.Layer):
                input = module(input, context)
            else:
                input = module(input)
            if name:
                context[name] = input
        return input


