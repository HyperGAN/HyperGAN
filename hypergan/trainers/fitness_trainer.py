import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class FitnessTrainer(BaseTrainer):
    def create(self):
        self.hist = [0 for i in range(2)]
        self.steps_since_fit=0
        config = self.config
        lr = config.learn_rate
        self.global_step = tf.train.get_global_step()
        decay_function = config.decay_function
        if decay_function:
            print("!!using decay function", decay_function)
            decay_steps = config.decay_steps or 50000
            decay_rate = config.decay_rate or 0.9
            decay_staircase = config.decay_staircase or False
            self.lr = decay_function(lr, self.global_step, decay_steps, decay_rate, decay_staircase)
        else:
            self.lr = lr

        return self._create()


    def _create(self):
        gan = self.gan
        config = self.config

        d_vars = self.d_vars or gan.discriminator.variables()
        g_vars = self.g_vars or (gan.encoder.variables() + gan.generator.variables())

        loss = self.loss or gan.loss
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
        reg = 0.5 * sum(
            tf.reduce_sum(tf.square(g)) for g in grads if g is not None
        )
        if config.update_rule == "ttur" or config.update_rule == 'single-step':
            Jgrads = [0 for i in allvars]
        else:
            Jgrads = tf.gradients(reg, allvars)

        print("JG", Jgrads)

        self.g_gradient = tf.ones([1])
        def amp_for(v):
            if v in g_vars:
                return config.g_w_lambda or 3
            if v in d_vars:
                return config.d_w_lambda or 1

        def applyvec(g, jg, v, decay):
            prev = v
            print("V", v,g,jg)
            nextw = v+g + jg * (config.jg_alpha or 0.1)
            if decay is not None:
                return ((decay) * prev + (1.0-decay)*nextw)-v
            else:
                return nextw-v

        def gradient_for(g, jg, v, decay):
            if config.update_rule == "ttur":
                if decay is not None:
                    amp = v+amp_for(v)*g
                    ng = ((decay) * v + (1.0-decay)*amp)-v
                else:
                    ng = amp_for(v)*g
            else:
                if decay is not None:
                    if v in g_vars:
                        ng = applyvec(g, jg, v, decay)
                    else:
                        ng = applyvec(g, jg, v, None)
                else:
                    ng = applyvec(g, jg, v, decay)
            return ng
        decay = config.g_exponential_moving_average_decay
        apply_vec = [ (gradient_for(g, Jg, v, decay), v) for (g, Jg, v) in zip(grads, Jgrads, allvars) if Jg is not None ]
        apply_vec_d = [ (gradient_for(g, Jg, v, decay), v) for (g, Jg, v) in zip(d_grads, Jgrads[:len(d_vars)], d_vars) if Jg is not None ]
        apply_vec_g = [ (gradient_for(g, Jg, v, decay), v) for (g, Jg, v) in zip(g_grads, Jgrads[len(d_vars):], g_vars) if Jg is not None ]

        defn = {k: v for k, v in config.items() if k in inspect.getargspec(config.trainer).args}
        tr = config.trainer(self.lr, **defn)


        optimizer = tr.apply_gradients(apply_vec, global_step=self.global_step)
        d_optimizer = tr.apply_gradients(apply_vec_d, global_step=self.global_step)
        g_optimizer = tr.apply_gradients(apply_vec_g, global_step=self.global_step)

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

        self.optimizer = optimizer
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.min_fitness=None
        
        if config.fitness_test is not None:
            mean = tf.zeros([1])
            if config.d_grads == "all":
                used_grads = grads
            elif config.d_grads:
                used_grads = d_grads
            else:
                used_grads = g_grads
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
            elif(config.fitness_type == 'ls3'):
                self.g_fitness = 1-loss.d_fake
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

        if config.g_ema_decay is not None:
            decay2 = config.g_ema_decay
            pg_vars = [tf.zeros_like(v) for v in g_vars]
            self.pg_vars = pg_vars
            self.g_vars = g_vars
            g_emas = [tf.assign(v, (decay2*pv+(1.0-decay2)*v)) for v, pv in zip(g_vars, pg_vars)]
            self.g_ema = tf.group(g_emas)

        return optimizer, optimizer

    def required(self):
        return "trainer learn_rate".split()

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = self.loss or gan.loss
        metrics = loss.metrics
        
        if config.g_ema_decay is not None:
            prev = sess.run(self.g_vars)
        if config.fitness_test is not None:
            self.steps_since_fit+=1
            if self.min_fitness is not None and np.isnan(self.min_fitness):
                print("NAN min fitness")
                self.min_fitness=None
            
            gl, dl, fitness,mean, *zs = sess.run([self.g_loss, self.d_loss, self.g_fitness, self.mean]+gan.fitness_inputs())
            g = None
            if(self.min_fitness is None or fitness < self.min_fitness or self.steps_since_fit > 1000):
                self.hist[0]+=1
                self.min_fitness = fitness
                self.steps_since_fit=0

                for v, t in ([[gl, self.g_loss],[dl, self.d_loss],[fitness, self.g_fitness]] + [ [v, t] for v, t in zip(zs, gan.fitness_inputs())]):
                    feed_dict[t]=v
                _, *metric_values = sess.run([self.optimizer] + self.output_variables(metrics), feed_dict)
            else:
                self.hist[1]+=1
                fitness_decay = config.fitness_decay or 0.99
                self.min_fitness = fitness_decay*self.min_fitness + (1.001-fitness_decay)*fitness
                if(config.train_d_on_fitness_failure):
                    metric_values = sess.run([self.d_optimizer]+self.output_variables(metrics), feed_dict)[1:]
                else:
                    metric_values = sess.run(self.output_variables(metrics), feed_dict)
                self.current_step-=1
        else:
            #standard
            metric_values = sess.run([self.optimizer] + self.output_variables(metrics), feed_dict)[1:]
        if config.g_ema_decay is not None:
            feed_dict = {}
            for p,pvalue in zip(self.pg_vars, prev):
                feed_dict[p]=pvalue
            _ = sess.run(self.g_ema, feed_dict)

        if self.current_step % 100 == 0:
            hist_output = "  " + "".join(["G"+str(i)+":"+str(v)+" "for i, v in enumerate(self.hist)])
            print(str(self.output_string(metrics) % tuple([self.current_step] + metric_values)+hist_output))
            self.hist = [0 for i in range(2)]

