import torch.nn as nn
import hypergan as hg
from hypergan.layer_size import LayerSize
from hypergan.modules.multi_head_attention import MultiHeadAttention as module_MultiHeadAttention

class MultiHeadAttention(hg.Layer):
    """
        ---
        description: 'layer multi_head_attention for configurable component'
        ---

        # multi_head_attention layer

        Adds an attention layer with multiple heads. Uses softmax. Ends in a linear

        ## required arguments

            `size` - output size

        ## optional arguments

            `heads` - Number of heads to use. Defaults to 4

        ## input size

        Any 2-d tensor

        ## output size

        First argument

        ## syntax

        ```json
          "ez_norm CHANNELS heads=HEADS"
        ```

        ## examples

        ```json
          "multi_head_attention 1024 heads=4"
        ```
    """
    def __init__(self, component, args, options):
        super(MultiHeadAttention, self).__init__(component, args, options)
        self.size = component.current_size.size()
        if len(args) > 0:
            self.size = args[0]
        self.layer = module_MultiHeadAttention(component.current_size.size(), self.size, heads=options.heads or 4)
        component.nn_init(self.layer.o, options.initializer)
        component.nn_init(self.layer.h, options.initializer)
        component.nn_init(self.layer.g, options.initializer)
        component.nn_init(self.layer.f, options.initializer)

    def output_size(self):
        return LayerSize(self.size)

    def forward(self, input, context):
        return self.layer.forward(input)


