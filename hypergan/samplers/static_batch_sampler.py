from hypergan.samplers.base_sampler import BaseSampler
import numpy as np

class StaticBatchSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.latent = self.gan.latent.next().data.clone()

    def compatible_with(gan):
        if hasattr(gan, 'latent'):
            return True
        return False

    def _sample(self):
        if(self.x is None):
            self.x = self.gan.inputs.next()
        self.gan.latent.z = self.latent
        g = self.gan.generator.forward(self.latent)
        samples = [
            ('generator', g),
        ]
        if hasattr(self.gan, 'g_to_z'):
            samples.append(('decoded', self.gan.generator(self.gan.g_to_z(g))))
            reconstruct_source_x = self.gan.generator(self.gan.g_to_z(self.gan.source)).clone().detach()
            samples.append(('source', self.gan.source))
            samples.append(('decodedx', reconstruct_source_x))
        if hasattr(self.gan, 'mask_layers'):
            for mask_layer in self.gan.mask_layers:
                samples.append(('mask_layer', mask_layer))
        if hasattr(self.gan, 'encoder'):
            encoding = self.gan.encoder(self.x)
            encoded_x = self.gan.decoder.forward(encoding)
            samples.append(('x', self.x))
            samples.append(('ex', encoded_x))


        return samples
