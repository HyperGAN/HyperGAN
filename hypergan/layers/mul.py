import hypergan as hg
from hypergan.layer_shape import LayerShape
from . import Operation

class Mul(Operation):
    """
        ---
        description: 'layer mul for configurable component'
        ---

        # mul layer

        Multiplies two or more layers together. Accepts nested layer definitions.

        ## input size

        Any number of matching tensors

        ## output size

        Same as input size

        ## syntax

        ```json
          "mul [layer]*"
        ```

        ## examples

        ```json
          "mul self (noise)"
        ```
    """


    def __init__(self, component, args, options):
        super(Mul, self).__init__("*", component, args, options)

