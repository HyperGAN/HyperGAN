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

        return self._create()

    def _create(self):
        gan = self.gan
        config = self.config

        d_vars = gan.d_vars()
        g_vars = gan.g_vars()

        d_vars = list(set(d_vars).intersection(tf.trainable_variables()))
        g_vars = list(set(g_vars).intersection(tf.trainable_variables()))

        loss = self.gan.loss
        d_loss, g_loss = loss.sample

        allloss = d_loss + g_loss

        allvars = d_vars + g_vars

        d_grads = tf.gradients(d_loss, d_vars)
        g_grads = tf.gradients(g_loss, g_vars)


        grads = d_grads + g_grads

        self.d_log = -tf.log(tf.abs(d_loss+TINY))
        for g, d_v in zip(grads,d_vars):
            if g is None:
                print("!!missing gradient")
                print(d_v)
                return
        apply_vec = []
        apply_vec_d = []
        apply_vec_g = []
        for (i, grad, v) in zip(range(len(grads)), grads, allvars): 

            if grad == None:
                print("WARNING: grad none", grad, v)
            else:
                apply_vec.append((grad, v))
                if v in d_vars:
                    apply_vec_d.append((grad, v))
                else:
                    apply_vec_g.append((grad, v))

        optimizer = hc.lookup_functions(config.optimizer)
        optimizer['gan']=self.gan
        optimizer['config']=optimizer
        defn = {k: v for k, v in optimizer.items() if k in inspect.getargspec(optimizer['class']).args}
        lr = optimizer.learn_rate or optimizer.learning_rate
        if 'learning_rate' in optimizer:
            del defn['learning_rate']
        tr = optimizer['class'](lr, **defn)

        self.gan.trainer = self
        self.g_loss = g_loss
        self.d_loss = d_loss

        self.gan.optimizer = tr

        optimize_t = tr.apply_gradients(apply_vec, global_step=self.global_step)
        d_optimize_t = tr.apply_gradients(apply_vec_d, global_step=self.global_step)

        self.past_weights = []

        self.g_loss = g_loss
        self.d_loss = d_loss
        self.slot_vars = tr.variables()

            
        def _slot_var(x, g_vars):
            for g in g_vars:
                if x.name.startswith(g.name.split(":")[0]):
                    return True
            return False
        self.slot_vars_g = [x for x in self.slot_vars if _slot_var(x, g_vars)]
        self.slot_vars_d = [x for x in self.slot_vars if _slot_var(x, d_vars)]

        self.optimize_t = optimize_t
        self.d_optimize_t = d_optimize_t
        self.min_fitness=None
        
        print("CONFIG ", config)
        if config.fitness_type is not None:
            mean = tf.zeros([1])
            used_grads = d_grads
            if config.grad_type == "sum":
                for g in used_grads:
                    mean += tf.reduce_sum(tf.abs(g))
            else:
                for g in used_grads:
                    mean += tf.reduce_mean(tf.abs(g))
                mean/=len(used_grads)
            self.mean=mean
            #self.mean=mean*100
            if config.fitness_type == 'g_loss':
                self.g_fitness = g_loss - (config.diversity_importance or 1) * tf.log(tf.abs(self.mean + d_loss - g_loss))
            elif(config.fitness_type == 'gradient-only'):
                self.g_fitness = -tf.log(reg)
            elif(config.fitness_type == 'grads'):
                self.g_fitness = mean
            elif(config.fitness_type == 'point'):
                self.g_fitness = mean - 1000*d_loss + 1000*g_loss
            elif(config.fitness_type == 'fail'):
                self.g_fitness = -mean
            elif(config.fitness_type == 'fail2'):
                self.g_fitness = -loss.d_fake
            elif(config.fitness_type == 'fail3'):
                self.g_fitness = -g_loss
            elif(config.fitness_type == 'fail2-reverse'):
                self.g_fitness = loss.d_fake
            elif(config.fitness_type == 'ls'):
                a,b,c = loss.config.labels
                self.g_fitness = tf.square(loss.d_fake-a)
            elif(config.fitness_type == 'ls-r'):
                a,b,c = loss.config.labels
                self.g_fitness = -tf.square(loss.d_fake-a)
            elif(config.fitness_type == 'ls2'):
                a,b,c = loss.config.labels
                self.g_fitness = tf.square(loss.d_fake-c)
            elif(config.fitness_type == 'ls2-r'):
                a,b,c = loss.config.labels
                self.g_fitness = -tf.square(loss.d_fake-c)
            elif(config.fitness_type == 'std'):
                self.g_fitness = -tf.nn.sigmoid(loss.d_fake)
            elif(config.fitness_type == 'ls3'):
                self.g_fitness = 1-loss.d_fake
            elif(config.fitness_type == 'ls4'):
                self.g_fitness = loss.d_real-loss.d_fake
            elif(config.fitness_type == 'ls5'):
                self.g_fitness = tf.square(loss.d_real)-tf.square(loss.d_fake)
            elif(config.fitness_type == 'fq1'):
                lam = 0.1
                self.g_fitness = -loss.d_fake-lam*mean
            elif(config.fitness_type == 'fq2'):
                lam = 0.1
                self.g_fitness = loss.d_real-loss.d_fake-lam*mean
            elif(config.fitness_type == 'fq3'):
                lam = 1
                self.g_fitness = loss.d_real-loss.d_fake+lam*mean
            elif(config.fitness_type == 'fq4'):
                lam = 1
                self.g_fitness = -loss.d_fake+lam*mean
            elif(config.fitness_type == 'fq5'):
                lam = 1
                self.g_fitness = -loss.d_fake-lam*tf.norm(mean)
            elif(config.fitness_type == 'fq6'):
                lam = 0.1
                self.g_fitness = -loss.d_fake-lam*tf.norm(mean+d_loss)
            elif(config.fitness_type == 'fq7'):
                lam = 0.1
                self.g_fitness = -loss.d_fake-lam*tf.norm(-mean-d_loss)
            elif(config.fitness_type == 'fq8'):
                lam = 0.1
                self.g_fitness = -tf.norm(mean+d_loss)
            elif(config.fitness_type == 'fq9'):
                lam = 0.1
                self.g_fitness = lam*mean
            elif(config.fitness_type == 'fq10'):
                lam = 0.1
                self.g_fitness = tf.norm(mean+d_loss)
            elif(config.fitness_type == 'fq11'):
                lam = 100.00
                self.fq = -loss.d_fake
                self.fd = lam * mean
                self.g_fitness = -loss.d_fake + lam * mean
            elif(config.fitness_type == 'ls3-fail'):
                self.g_fitness = -(1-loss.d_fake)
            elif(config.fitness_type == 'gldl'):
                self.g_fitness = -d_loss + g_loss
            elif(config.fitness_type == 'df'):
                self.g_fitness = tf.abs(loss.d_fake) - tf.abs(loss.d_real)
            elif(config.fitness_type == 'standard'):
                self.g_fitness = tf.reduce_mean(g_loss) - (config.diversity_importance or 1)* tf.log(tf.abs(self.mean - tf.log(TINY+tf.sigmoid(d_loss)) - \
                        tf.log(1.0-tf.sigmoid(g_loss)+TINY)))
            else:
                self.g_fitness = tf.reduce_mean(loss.d_fake) - (config.diversity_importance or 1)* tf.log(tf.abs(self.mean + tf.reduce_mean(loss.d_real) - tf.reduce_mean(loss.d_fake)))
            self.g_fitness = tf.reduce_mean(self.g_fitness)

        return optimize_t, optimize_t

    def required(self):
        return "".split()

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = self.loss 
        metrics = gan.metrics()

        feed_dict = {}

        fit = False
        steps_since_fit = 0
        old_fitness = None
        while not fit:
            steps_since_fit+=1
            gl, dl, fitness, *zs = sess.run([self.g_loss, self.d_loss, self.g_fitness]+gan.fitness_inputs())
            if np.isnan(fitness) or np.isnan(gl) or np.isnan(dl):
                print("NAN Detected.  Candidate done")
                self.min_fitness = None
                self.mix_threshold_reached = True
                return
            if old_fitness == fitness:
                print("Stuck state detected, unsticking")
                self.min_fitness = None
                return
            old_fitness = fitness


            g = None
            if self.config.skip_fitness:
                self.min_fitness = None
            if(self.min_fitness is None or fitness <= self.min_fitness):
                self.hist[0]+=1
                self.min_fitness = fitness
                steps_since_fit=0
                if config.assert_similarity:
                    if((gl - dl) > ((config.similarity_ratio or 1.8) * ( (gl + dl) / 2.0)) ):
                        print("g_loss - d_loss > allowed similarity threshold", gl, dl, gl-dl)
                        self.min_fitness = None
                        self.mix_threshold_reached = True
                        return

                for v, t in ([[gl, self.g_loss],[dl, self.d_loss]] + [ [v, t] for v, t in zip(zs, gan.fitness_inputs())]):
                    feed_dict[t]=v

                for i in range(self.config.d_update_steps or 0):
                    self.before_step(self.current_step, feed_dict)
                    _, *metric_values = sess.run([self.d_optimize_t], feed_dict)
                    self.after_step(self.current_step, feed_dict)

                self.before_step(self.current_step, feed_dict)
                _, *metric_values = sess.run([self.optimize_t] + self.output_variables(metrics), feed_dict)
                self.after_step(self.current_step, feed_dict)
                fit=True
            else:
                self.hist[1]+=1
                fitness_decay = config.fitness_decay or 0.99
                self.min_fitness = self.min_fitness + (1.00-fitness_decay)*(fitness-self.min_fitness)
                metric_values = sess.run(self.output_variables(metrics), feed_dict)

            steps_since_fit=0

        if ((self.current_step % 10) == 0):
            hist_output = "  " + "".join(["G"+str(i)+":"+str(v)+" "for i, v in enumerate(self.hist)])
            print(str(self.output_string(metrics) % tuple([self.current_step] + metric_values)+hist_output))
            self.hist = [0 for i in range(2)]

