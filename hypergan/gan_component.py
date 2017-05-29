import hyperchamber as hc


class ValidationException(Exception):
    pass

class GANComponent:
    def __init__(self, config):
        self.config = hc.Config(config)
        errors = self.validate()
        if errors != []:
            raise ValidationException(",".join(errors))

    def validate(self):
        return []

