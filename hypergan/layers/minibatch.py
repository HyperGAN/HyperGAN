import hypergan as hg
from hypergan.layer_shape import LayerShape

class Minibatch(hg.Layer):
    """A minimal Minibatch layer used by the MinibatchTrainHook.

    This is a small, safe implementation that passes inputs through and
    advertises the same output shape as the input. The real implementation
    is TensorFlow-specific; adding this here prevents import-time failures
    when running tests in environments that don't exercise the TensorFlow
    code paths.
    """
    def __init__(self, component, args, options):
        super(Minibatch, self).__init__(component, args, options)
        self.dims = list(component.current_size.dims).copy()

    def output_size(self):
        return LayerShape(*self.dims)

    def forward(self, input, context):
        # Pass-through implementation
        return input
