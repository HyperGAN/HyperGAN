import hypergan as hg
from hypergan.layer_size import LayerSize
from . import Operation

class Add(Operation):
    def __init__(self, component, args, options):
        super(Add, self).__init__("+", component, args, options)

