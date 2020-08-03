import hypergan as hg
from hypergan.layer_shape import LayerShape
from . import Operation

class Cat(Operation):
    """
        ---
        description: 'layer cat for configurable component'
        ---

        # cat layer

        Concatenate two or more layers together. Accepts nested layer definitions.

        ## input size

        Any number of matching tensors

        ## output size

        Same as input size

        ## syntax

        ```json
          "cat [layer]*"
        ```

        ## examples

        ```json
          "cat self (attention)"
        ```
    """

    def __init__(self, component, args, options):
        super(Cat, self).__init__("cat", component, args, options)
        self.size = LayerShape(sum([layer.channels for layer in self.layer_sizes]), *self.size.dims[1:])

