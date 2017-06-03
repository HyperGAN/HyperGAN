import hyperchamber as hc
import inspect


class ValidationException(Exception):
    pass

class GANComponent:
    def __init__(self, gan, config):
        self.gan = gan
        self.config = hc.Config(config)
        errors = self.validate()
        if errors != []:
            raise ValidationException("\n".join(errors))
        self.create_ops(config)

    def create_ops(self, config):
        if self.gan is None:
            return
        if self.gan.ops_backend is None:
            return
        filtered_options = {k: v for k, v in config.items() if k in inspect.getargspec(self.gan.ops_backend).args}
        self.ops = self.gan.ops_backend(*dict(filtered_options))
        self.config = self.ops.lookup(config)

    def required(self):
        return []

    def validate(self):
        errors = []
        required = self.required()
        for argument in required:
            if(self.config.__getattr__(argument) == None):
                errors.append("`"+argument+"` required")

        if(self.gan is None):
            errors.append("GANComponent constructed without GAN")
        return errors

    def weights(self):
        return self.ops.weights

    def biases(self):
        return self.ops.biases

    def variables(self):
        return self.ops.variables()

