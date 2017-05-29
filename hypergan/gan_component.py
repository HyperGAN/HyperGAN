import hyperchamber as hc
import inspect


class ValidationException(Exception):
    pass

class GANComponent:
    def __init__(self, gan, config):
        self.config = hc.Config(config)
        self.gan = gan
        self.ops = self.create_ops()
        errors = self.validate()
        if errors != []:
            raise ValidationException("\n".join(errors))

    def create_ops(self):
        filtered_options = {k: v for k, v in self.config.items() if k in inspect.getargspec(self.gan.ops).args}
        ops = self.gan.ops(*dict(filtered_options))
        return ops

    def required(self):
        return []

    def validate(self):
        errors = []
        required = self.required()
        for argument in required:
            if(self.config.__getattr__(argument) == None):
                errors.append("`"+argument+"` required")

        return errors


