import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class BatchFitnessTrainer(BaseTrainer):
    def create(self):
        self.hist = [0 for i in range(2)]
        config = self.config
        self.global_step = tf.train.get_global_step()
        self.mix_threshold_reached = False
        decay_function = config.decay_function
        self.min_fitness = None
        super(BatchFitnessTrainer, self).create()

    def _create(self):
        gan = self.gan
        config = self.config
        loss = gan.loss
        d_vars = gan.d_vars()
        g_vars = gan.g_vars()
        self._delegate = self.gan.create_component(config.trainer)
        ftype = config.type

        self.fitness = -loss.d_fake
        if self.config.reverse:
            self.fitness = loss.d_fake
        if self.config.abs:
            self.fitness = tf.abs(self.gan.loss.d_fake)
        if self.config.nabs:
            self.fitness = -tf.abs(self.gan.loss.d_fake)
        self.zs = None

    def variables(self):
        return self._delegate.variables()

    def required(self):
        return "".split()

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = self.gan.loss 
        metrics = gan.metrics()

        feed_dict = {}

        fit = False
        if self.zs is None:
            d = sess.run([self.fitness,gan.latent.sample])
            fitness = d[0]
            zs = d[1]
        else:
            fitness = sess.run(self.fitness)
            zs = self.zs
        if self.config.heuristic is not None:
            last_fitness = 10000
            count = 0
            for i in range(self.config.search_steps or 2):
                d = sess.run([self.fitness,gan.latent.sample])
                _f = d[0]
                _z = d[1]
                fitness = np.reshape(fitness, np.shape(_f))
                fitness = np.concatenate([fitness,_f], axis=0)
                zs = np.reshape(zs, np.shape(_z))
                zs = np.concatenate([zs,_z], axis=0)
                sort = np.argsort(fitness.flatten())[:gan.batch_size()]
                zs = zs[sort]
                fitness = fitness.flatten()[sort]
                if fitness.flatten()[-1] < last_fitness:
                    last_fitness = fitness[-1]
                    count = 0
                else:
                    count += 1
                    if count > self.config.heuristic:
                        #print("z fit ", i)
                        sort_zs = np.reshape(zs, np.shape(_z))
                        break
        else:
            for i in range(self.config.search_steps or 2):
                d = sess.run([self.fitness,gan.latent.sample])
                _f = d[0]
                _z = d[1]
                fitness = np.concatenate([fitness,_f], axis=0)
                zs = np.concatenate([zs,_z], axis=0)
            fitness = np.array(fitness).flatten()
            sort = np.argsort(fitness)
            sort = sort[:gan.batch_size()]
            sort_zs = zs[sort]
        feed_dict[gan.latent.sample]=sort_zs
        self.zs = sort_zs
        
        self.before_step(self.current_step, feed_dict)
        self._delegate.step(feed_dict)

        self.after_step(self.current_step, feed_dict)

