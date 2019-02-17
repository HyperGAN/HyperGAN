import numpy as np
import tensorflow as tf
from PIL import Image
from hypergan.viewer import GlobalViewer
from hypergan.samplers.base_sampler import BaseSampler
import time

class BatchWalkSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8, session=None):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.z_start = None
        self.y = None
        self.x = None
        self.step = 0
        self.steps = []
        self.step_count = 30
        self.target = None
        self.rows = 2
        self.columns = 4
        self.needed = int(self.rows*self.columns / gan.batch_size())
        #self.style_t = gan.styleb.sample
        #self.style_v = gan.session.run(self.style_t)


    def regenerate_steps(self):
        gan = self.gan
        z_t = gan.latent.sample
        inputs_t = gan.inputs.x

        s=np.shape(gan.session.run(gan.latent.sample))
        bs = 16
        if self.z_start is None:
            self.z_start = [gan.session.run(gan.latent.sample)[0] for _ in range(self.needed)]
        else:
            print("UPDAING Z STARCT")
            self.z_start = [self.steps[i][-1] for i in range(len(self.z_start))]

        targ = [gan.session.run(gan.latent.sample)[0] for _ in range(self.needed)]

        z_interps = []
        for i in range(len(self.z_start)):
            mask = np.linspace(0., 1., num=bs)
            z = np.tile(np.expand_dims(np.reshape(self.z_start[i], [-1]), axis=0), [bs,1])
            tg = np.tile(np.expand_dims(np.reshape(targ[i], [-1]), axis=0), [bs,1])
            mask = np.tile(np.expand_dims(mask, axis=1), [1, np.shape(self.z_start[i])[-1]])
            z_interp = np.multiply(1.0-mask,z) + mask*tg
            z_interps += [z_interp]
        print('z_INT', np.shape(z_interps))
        return z_interps

    def _sample(self):
        gan = self.gan
        z_t = gan.latent.sample
        inputs_t = gan.inputs.x
        self.step+=1

        if(len(self.steps) == 0 or self.step >= len(self.steps[0])):
            self.steps = self.regenerate_steps()
            self.step=0

        gs = []
        for i in range(int(self.needed)):
            z = self.steps[i][self.step]
            z = np.expand_dims(z,axis=0)
            g = gan.session.run(gan.generator.sample, feed_dict={z_t: z})
            gs.append(g)
        g = np.hstack(gs)
        xshape = gan.ops.shape(gan.inputs.x)
        g = np.reshape(gs, [self.rows, self.columns, xshape[1], xshape[2], xshape[3]])
        g = np.concatenate(g, axis=1)
        g = np.concatenate(g, axis=1)
        g = np.expand_dims(g, axis=0)
        return {
            'generator': g
        }

    def compatible_with(gan):
        if hasattr(gan, 'latent') and gan.batch_size() == 1:
            return True
        return False


    def plot(self, image, filename, save_sample):
        """ Plot an image."""
        image = np.minimum(image, 1)
        image = np.maximum(image, -1)
        image = np.squeeze(image)
        # Scale to 0..255.
        imin, imax = image.min(), image.max()
        image = (image - imin) * 255. / (imax - imin) + .5
        image = image.astype(np.uint8)
        if save_sample:
            try:
                Image.fromarray(image).save(filename)
            except Exception as e:
                print("Warning: could not sample to ", filename, ".  Please check permissions and make sure the path exists")
                print(e)
        GlobalViewer.update(self.gan, image)
