import hyperchamber as hc


class ValidationException(Exception):
    pass

class GANComponent:
    def __init__(self, config):
        self.config = hc.Config(config)
        errors = self.validate()
        if errors != []:
            raise ValidationException("\n".join(errors))

    def required(self):
        return []

    def validate(self):
        errors = []
        required = self.required()
        for argument in required:
            if(self.config.__getattr__(argument) == None):
                errors.append("`"+argument+"` required")

        return errors


