import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class FitnessTrainer(BaseTrainer):
    def create(self):
        self.hist = [0 for i in range(2)]
        config = self.config
        self.global_step = tf.train.get_global_step()
        self.mix_threshold_reached = False
        decay_function = config.decay_function
        self.min_fitness = None

        super(FitnessTrainer, self).create()

    def _create(self):
        gan = self.gan
        config = self.config
        loss = gan.loss
        d_vars = gan.d_vars()
        g_vars = gan.g_vars()
        self._delegate = self.gan.create_component(config.trainer)
        ftype = config.type

        if(ftype == 'fail2'):
            self.fitness = -loss.d_fake
        elif(ftype == 'fail2-reverse'):
            self.fitness = loss.d_fake
        elif(ftype == 'ls'):
            a,b,c = loss.config.labels
            self.fitness = tf.square(loss.d_fake-a)
        elif(ftype == 'ls-r'):
            a,b,c = loss.config.labels
            self.fitness = -tf.square(loss.d_fake-a)
        elif(ftype == 'ls2'):
            a,b,c = loss.config.labels
            self.fitness = tf.square(loss.d_fake-c)
        elif(ftype == 'ls2-r'):
            a,b,c = loss.config.labels
            self.fitness = -tf.square(loss.d_fake-c)
        elif(ftype == 'std'):
            self.fitness = -tf.nn.sigmoid(loss.d_fake)
        elif(ftype == 'ls3'):
            self.fitness = 1-loss.d_fake
        elif(ftype == 'ls4'):
            self.fitness = loss.d_real-loss.d_fake
        elif(ftype == 'ls5'):
            self.fitness = tf.square(loss.d_real)-tf.square(loss.d_fake)
        elif(ftype == 'fq1'):
            lam = 0.1
            self.fitness = -loss.d_fake-lam*mean
        elif(ftype == 'fq2'):
            lam = 0.1
            self.fitness = loss.d_real-loss.d_fake-lam*mean
        elif(ftype == 'fq3'):
            lam = 1
            self.fitness = loss.d_real-loss.d_fake+lam*mean
        elif(ftype == 'fq4'):
            lam = 1
            self.fitness = -loss.d_fake+lam*mean
        elif(ftype == 'fq5'):
            lam = 1
            self.fitness = -loss.d_fake-lam*tf.norm(mean)
        elif(ftype == 'fq6'):
            lam = 0.1
            self.fitness = -loss.d_fake-lam*tf.norm(mean+d_loss)
        elif(ftype == 'fq7'):
            lam = 0.1
            self.fitness = -loss.d_fake-lam*tf.norm(-mean-d_loss)
        elif(ftype == 'fq8'):
            lam = 0.1
            self.fitness = -tf.norm(mean+d_loss)
        elif(ftype == 'fq9'):
            lam = 0.1
            self.fitness = lam*mean
        elif(ftype == 'fq10'):
            lam = 0.1
            self.fitness = tf.norm(mean+d_loss)
        elif(ftype == 'fq11'):
            lam = 100.00
            self.fq = -loss.d_fake
            self.fd = lam * mean
            self.fitness = -loss.d_fake + lam * mean
        elif(ftype == 'ls3-fail'):
            self.fitness = -(1-loss.d_fake)
        elif(ftype == 'gldl'):
            self.fitness = -d_loss + g_loss
        elif(ftype == 'df'):
            self.fitness = tf.abs(loss.d_fake) - tf.abs(loss.d_real)
        elif(ftype == 'standard'):
            self.fitness = tf.reduce_mean(g_loss) - (config.diversity_importance or 1)* tf.log(tf.abs(self.mean - tf.log(TINY+tf.sigmoid(d_loss)) - \
                    tf.log(1.0-tf.sigmoid(g_loss)+TINY)))
        else:
            #self.fitness = tf.reduce_mean(loss.d_fake) - (config.diversity_importance or 1)* tf.log(tf.abs(self.mean + tf.reduce_mean(loss.d_real) - tf.reduce_mean(loss.d_fake)))
            self.fitness = -loss.d_fake
        self.fitness = tf.reduce_mean(self.fitness)

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
        steps_since_fit = 0
        old_fitness = None
        while not fit:
            steps_since_fit+=1
            fitness, *zs = sess.run([self.fitness,gan.latent.sample])
            if old_fitness == fitness:
                print("Stuck state detected, unsticking")
                self.min_fitness = None
                return
            old_fitness = fitness


            g = None
            if(self.min_fitness is None or fitness <= self.min_fitness):
                self.hist[0]+=1
                self.min_fitness = fitness
                steps_since_fit=0

                for v, t in ([ [v, t] for v, t in zip(zs, [gan.latent.sample])]):
                    feed_dict[t]=v

                self.before_step(self.current_step, feed_dict)
                self._delegate.step(feed_dict)

                self.after_step(self.current_step, feed_dict)
                fit=True
            else:
                self.hist[1]+=1
                fitness_decay = config.fitness_decay or 0.99
                self.min_fitness = self.min_fitness + (1.00-fitness_decay)*(fitness-self.min_fitness)

            steps_since_fit=0

        if ((self.current_step % 10) == 0):
            hist_output = "  " + "".join(["G"+str(i)+":"+str(v)+" "for i, v in enumerate(self.hist)])
            print(hist_output)
            self.hist = [0 for i in range(2)]

