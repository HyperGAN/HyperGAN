import hypergan as hg
from hypergan.layer_size import LayerSize
from . import Operation

class Mul(Operation):
    def __init__(self, component, args, options):
        super(Mul, self).__init__("*", component, args, options)

