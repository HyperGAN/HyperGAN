from hypergan.samplers.base_sampler import BaseSampler

class InputSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)

    def _sample(self):
        return {
            'generator': self.gan.inputs.next()[0]
        }

