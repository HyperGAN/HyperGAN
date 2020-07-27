import hypergan as hg
from hypergan.layer_size import LayerSize
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

    #TODO custom output_size
    #def layer_cat(self, net, args, options):
    #    result = self.operation_layer(net, args, options, "cat")
    #    dims = None
    #    for arg in args:
    #        if arg == "self":
    #            new_size = self.current_size
    #        else:
    #            new_size = self.layer_output_sizes[args[1]]
    #        if dims is None:
    #            dims = new_size.dims
    #        else:
    #            dims = [dims[0]+new_size.dims[0]] + list(dims[1:])

    #    self.current_size = LayerSize(*dims)
    #    return result


