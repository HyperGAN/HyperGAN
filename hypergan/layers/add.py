import hypergan as hg
from hypergan.layer_size import LayerSize
from . import Operation

class Add(Operation):
    """
        ---
        description: 'layer add for configurable component'
        ---

        # add layer

        Adds two or more layers together. Accepts nested layer definitions.

        ## input size

        Any number of matching tensors

        ## output size

        Same as input size

        ## syntax

        ```json
          "add [layer]*"
        ```

        ## examples

        ```json
          "add self (attention)"
        ```
    """
    def __init__(self, component, args, options):
        super(Add, self).__init__("+", component, args, options)

