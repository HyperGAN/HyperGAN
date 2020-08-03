import torch.nn as nn
import torch

import hypergan as hg
import hyperchamber as hc
import pyparsing

from hypergan.gan_component import ValidationException

class Operation(hg.Layer):
    """
        ---
        description: Base class for operations
        ---

        This is a base class for many operations and not used directly.
    """
    def __init__(self, operation, component, args, options):
        super(Operation, self).__init__(component, args, options)
        self.operation = operation
        self.size = component.current_size
        self.layers, self.layer_names, self.layer_sizes = self.build_layers(component, args, options)
        self.modules = nn.ModuleList(self.layers)

    def build_layers(self, component, args, options):
        options = hc.Config(options)
        layers = []
        layer_names = []
        layer_shapes = []

        for arg in args:
            component.current_size = self.size
            if arg == 'self':
                layers.append(None)
                layer_names.append("self")
                layer_shapes.append(self.size)
            elif arg == 'noise':
                layers.append(LearnedNoise())
                layer_names.append(None)
                layer_shapes.append(self.size)
            elif arg in component.named_layers:
                layers.append(None)
                layer_names.append("layer "+arg)
                layer_shapes.append(component.layer_output_sizes[arg])
            elif arg in component.gan.named_layers:
                layers.append(component.gan.named_layers[arg])
                layer_names.append(None)
                layer_shapes.append(component.layer_output_sizes[arg])
            elif arg in component.context_shapes:
                layers.append(None)
                layer_names.append(arg)
                layer_shapes.append(component.context_shapes[arg])
            elif type(arg) == pyparsing.ParseResults and type(arg[0]) == hg.parser.Pattern:
                parsed = arg[0]
                parsed.parsed_options = hc.Config(parsed.options)
                layer = component.build_layer(parsed.layer_name, parsed.args, parsed.parsed_options)
                layers.append(layer)
                layer_names.append(parsed.layer_name)
                layer_shapes.append(component.current_size)
            else:
                raise ValidationException("Could not parse operation layer '" + arg + "'")
        return layers, layer_names, layer_shapes

    def output_size(self):
        return self.size

    def forward(self, input, context):
        output = None
        for layer, layer_name in zip(self.layers, self.layer_names):
            if layer_name == "self":
                layer_output = input
            elif isinstance(layer, hg.Layer):
                layer_output = layer(input, context)
            elif layer_name.split(" ")[0] == 'layer':
                layer_output = context[layer_name.split(" ")[1]]
            elif layer_name in context:
                layer_output = context[layer_name]
            else:
                layer_output = layer(input)
            if output is None:
                output = layer_output
            else:
                if self.operation == "+":
                    output = output + layer_output
                elif self.operation == "*":
                    output = output * layer_output
                elif self.operation == "cat":
                    output = torch.cat([output, layer_output], 1)
                else:
                    raise ValidationException("Unknown operation: "+ self.operation)
        return output
