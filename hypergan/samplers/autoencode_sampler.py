from hypergan.samplers.base_sampler import BaseSampler
import tensorflow as tf
import numpy as np

class AutoencodeSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.z = None
        self.y = None
        self.x = None

    def _sample(self):
        gan = self.gan
        inputs_t = gan.inputs.x
        z_t = gan.encoder.sample

        if self.z is None:
            self.input = gan.session.run(inputs_t)
        self.z = gan.session.run(z_t, feed_dict={inputs_t: self.input})

        destination = self.z[1]
        origin = self.z[0]
        for i in range(0, np.shape(self.z)[0], self.samples_per_row):
            last = i+self.samples_per_row-1
            multiple = np.linspace(0, 1, self.samples_per_row-4)

            for j in range(i+2, last-1):
                percent = (j - (i))/float((last) - (i+1))
                self.z[j] = self.z[i]*(1.0-percent) + (self.z[last])*percent
            self.z[i+1] = self.z[i]
            self.z[last-1] = self.z[last]
 
        output = gan.session.run(gan.generator.sample, feed_dict={z_t: self.z})
        for i in range(0, np.shape(self.z)[0], self.samples_per_row):
            last = i+self.samples_per_row-1
            output[i] = self.input[i]
            output[last] = self.input[last] 
        

        g=tf.get_default_graph()
        with g.as_default():
            tf.set_random_seed(1)
            return {
                'generator': output
            }

