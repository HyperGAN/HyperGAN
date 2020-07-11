import hyperchamber as hc
import inspect
import itertools
import types
import importlib
import torch.nn as nn

class ValidationException(Exception):
    """
    GAN components validate their configurations before creation.  
    
    `ValidationException` occcurs if they fail.
    """
    pass

class GANComponent(nn.Module):
    """
    GANComponents are pluggable pieces within a GAN.

    GAN objects are also GANComponents.
    """
    def __init__(self, gan, config):
        """
        Initializes a gan component based on a `gan` and a `config` dictionary.

        Different components require different config variables.  

        A `ValidationException` is raised if the GAN component configuration fails to validate.
        """
        super(GANComponent, self).__init__()
        self.gan = gan
        self.config = hc.Config(config)
        errors = self.validate()
        if errors != []:
            raise ValidationException(self.__class__.__name__+": " +"\n".join(errors))
        self.create()

    def create(self, *args):
        raise ValidationException("GANComponent.create() called directly.  Please override.")

    def required(self):
        """
        Return a list of required config strings and a `ValidationException` will be thrown if any are missing.

        Example: 
        ```python
            class MyComponent(GANComponent):
                def required(self):
                    "learn rate is required"
                    ["learn_rate"]
        ```
        """
        return []

    def validate(self):
        """
        Validates a GANComponent.  Return an array of error messages. Empty array `[]` means success.
        """
        errors = []
        required = self.required()
        for argument in required:
            if(self.config.__getattr__(argument) == None):
                errors.append("`"+argument+"` required")

        if(self.gan is None):
            errors.append("GANComponent constructed without GAN")
        return errors

    def add_metric(self, name, value):
        """adds metric to monitor during training
            name:string
            value:Tensor
        """
        return self.gan.add_metric(name, value)

    def metrics(self):
        """returns a metric : tensor hash"""
        return self.gan.metrics()


    def layer_regularizer(self, net):
        symbol = self.config.layer_regularizer
        op = self.lookup_function(symbol)
        if op and isinstance(op, types.FunctionType):
            net = op(self, net)
        return net

    def lookup_function(self, name):
        namespaced_method = name.split(":")[1]
        method = namespaced_method.split(".")[-1]
        namespace = ".".join(namespaced_method.split(".")[0:-1])
        return getattr(importlib.import_module(namespace),method)

    def lookup_class(self, name):
        return self.lookup_function(name)

    def set_trainable(self, flag):
        for p in self.parameters():
            p.requires_grad = flag
