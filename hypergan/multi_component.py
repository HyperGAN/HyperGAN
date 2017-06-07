from hypergan.gan_component import GANComponent

import tensorflow as tf

class MultiComponent():
    """
        Used to combine multiple components into one.  For example, `gan.loss = MultiComponent([loss1, loss2])`
    """
    def __init__(self, components=[]):
        self.components = components

    def __getattr__(self, name):
        if len(self.components) == 0:
            return None

        attributes = self.lookup(name)
        return self.combine(attributes)

    def lookup(self, name):
        lookups = []
        for component in self.components:
            if hasattr(component, name):
                lookups.append(getattr(component,name))
            else:
                return None

        return lookups

    def combine(self, data):
        if data == None or data == []:
            return data

        if type(data[0]) == tf.Tensor:
            return self.components[0].ops.concat(values=data, axis=1)

        if callable(data[0]):
            return self.call_each(data)
        return data

    def call_each(self, methods):
        def do_call(*args, **kwargs):
            results = []
            for method in methods:
                results.append(method(*args, **kwargs))
            return self.combine(results)
        return do_call
