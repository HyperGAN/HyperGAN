from hypergan.samplers.base_sampler import BaseSampler
import tensorflow as tf
import numpy as np

class RandomWalkSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.assign = None
        self.z = None
        self.y = None
        self.x = None
        self.step = 0
        self.steps = 30
        self.target = None

    def _sample(self):
        gan = self.gan
        z_t = gan.encoder.sample
        inputs_t = gan.inputs.x

        if self.z is None:
            self.z = gan.encoder.sample.eval()
            #self.target = gan.encoder.sample.eval()
            self.target = np.random.uniform(-1,1,np.shape(self.z))
            self.input = gan.session.run(gan.inputs.x)

        if self.step > self.steps:
            self.z = self.target
            #self.target = gan.encoder.sample.eval()
            self.target = np.random.uniform(-1,1,np.shape(self.z))
            self.step = 0

        percent = float(self.step)/self.steps
        z_interp = self.z*(1.0-percent) + self.target*percent
        self.step+=1

        if self.gan.config.sampler_assign:
            #print("SAMPLAER ASSIGN", z_interp)
            self.assign = z_t.assign(z_interp)
            self.gan.session.run(self.assign)

        g=tf.get_default_graph()
        with g.as_default():
            tf.set_random_seed(1)
            return {
                'generator': gan.session.run(gan.generator.sample, feed_dict={z_t: z_interp, inputs_t: self.input})
            }

