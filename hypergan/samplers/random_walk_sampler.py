from hypergan.samplers.base_sampler import BaseSampler
import tensorflow as tf

class RandomWalkSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.z = None
        self.y = None
        self.x = None
        self.step = 0
        self.steps = 30
        self.target = None

    def _sample(self):
        gan = self.gan
        z_t = gan.encoder.sample #TODO
        inputs_t = gan.inputs.x

        if self.z is None:
            print("GAN IS", gan, gan.encoder)
            self.z = z_t.eval()
            self.input = gan.session.run(gan.inputs.x)

        if self.target is None or self.step > self.steps:
            self.target = gan.uniform_encoder.z.eval()
            self.step = 0

        z_interp = self.z + self.target*float(self.step)/self.steps
        self.step+=1
        print(self.step)

        g=tf.get_default_graph()
        with g.as_default():
            tf.set_random_seed(1)
            return {
                'generator': gan.session.run(gan.generator.sample, feed_dict={z_t: z_interp, inputs_t: self.input})
            }

