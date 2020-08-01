from hypergan.samplers.base_sampler import BaseSampler
import numpy as np
import torch


class GridSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, 8)

    def _sample(self):
        gan = self.gan
        y = np.linspace(0,1, 6)

        z = np.mgrid[-0.999:0.999:0.6, -0.999:0.999:0.26].reshape(2,-1).T
        z = np.reshape(z, [32,2])
        #z = np.mgrid[-0.499:0.499:0.3, -0.499:0.499:0.13].reshape(2,-1).T
        #z = np.mgrid[-0.299:0.299:0.15, -0.299:0.299:0.075].reshape(2,-1).T
        needed = 32 / gan.batch_size()
        gs = []
        for i in range(int(needed)):
            zi = z[i*gan.batch_size():(i+1)*gan.batch_size()]
            zi = torch.from_numpy(zi).type(torch.FloatTensor).cuda()
            g = gan.generator(zi).detach().cpu().numpy()
            gs.append(g)
        g = np.reshape(gs, [4, 8, gan.channels(), gan.height(), gan.width()])
        g = np.concatenate(g, axis=0)
        g = torch.from_numpy(g).cuda()
        #x_hat = gan.session.run(gan.autoencoded_x, feed_dict={gan.inputs.x: self.x})
        #e = gan.session.run(gan.encoder.sample, feed_dict={gan.inputs.x: g})

        return [
            ('generator',g)
            ]
