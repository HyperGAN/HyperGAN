import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect
import nashpy as nash
import hypergan as hg
import hyperchamber as hc
import sys
import gc
import os
import random

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class GangTrainer(BaseTrainer):
    def create(self):
        config = self.config
        gan = self.gan
        loss = self.gan.loss
        d_vars = gan.d_vars()
        g_vars = gan.g_vars()
        self.reinit = tf.variables_initializer(d_vars+g_vars)
        self.priority_ds = []
        self.priority_gs = []


        self._delegate = self.gan.create_component(config.rbbr, d_vars=d_vars, g_vars=g_vars)
        self.optimizer = self._delegate

        print("VARS  ___", self.optimizer.variables())
        for var in self.optimizer.variables():
            if "generator" in var.name:
                g_vars += [var]
            elif "discriminator" in var.name:
                d_vars += [var]
            else:
                print("[WARNING] Unknown gang variable, not added to g_vars or d_vars: ", var)
        d_loss, g_loss = loss.sample
        if self.config.fitness_method == "wasserstein":
            self.gang_loss = -(loss.d_real-loss.d_fake)
        elif self.config.fitness_method == "least_squares":
            b = gan.loss.config.labels[1]
            a = gan.loss.config.labels[0]
            self.gang_loss = tf.sign(loss.d_fake + loss.d_real) * tf.square(loss.d_fake+loss.d_real)
        elif self.config.fitness_method == "least_squares2":
            self.gang_loss = tf.square(loss.d_fake)-tf.square(loss.d_real)
        elif self.config.fitness_method == "least_squares3":
            c = gan.loss.config.labels[2]
            b = gan.loss.config.labels[1]
            a = gan.loss.config.labels[0]
            self.gang_loss = tf.square(loss.d_fake)*tf.sign(loss.d_fake) + tf.square(loss.d_real)*tf.sign(d_real)
        elif self.config.fitness_method == "least_squares4":
            c = gan.loss.config.labels[2]
            b = gan.loss.config.labels[1]
            a = gan.loss.config.labels[0]
            self.gang_loss = tf.square((-loss.d_fake+1)/2) - tf.square((-loss.d_real+1)/2)

        elif self.config.fitness_method == 'double':
            self.gang_loss = [loss.d_fake+loss.d_real, loss.d_real-loss.d_fake]
        elif self.config.fitness_method == 'double_ragan':
            self.gang_loss = [0.7*(loss.d_fake-loss.d_real)+0.3*(loss.d_real-loss.d_fake), 
                    0.7*(loss.d_real-loss.d_fake)-0.3*(loss.d_fake-loss.d_real)]
        elif self.config.fitness_method == 'zero_raganls':
            c = gan.loss.config.labels[2]
            b = gan.loss.config.labels[1]
            a = gan.loss.config.labels[0]
            self.gang_loss = tf.square(loss.d_real-loss.d_fake-b) - tf.square(loss.d_fake - loss.d_real - b)
        elif self.config.fitness_method == 'zero_raganls2':
            c = gan.loss.config.labels[2]
            b = gan.loss.config.labels[1]
            a = gan.loss.config.labels[0]
            self.gang_loss = tf.square(loss.d_real-loss.d_fake-b)
        elif self.config.fitness_method == '-gl-dl':
            self.gang_loss = [-g_loss, -d_loss]
        elif self.config.fitness_method == 'gl-dl2':
            self.gang_loss = [-g_loss, g_loss]
        elif self.config.fitness_method == 'gldl':
            # breaks
            self.gang_loss = [g_loss, d_loss]
        elif self.config.fitness_method == 'double2':
            a = gan.loss.config.labels[0]
            b = gan.loss.config.labels[1]
            self.gang_loss = [tf.square(loss.d_fake-a)+tf.square(loss.d_real-a), tf.square(los++d_real-a)+tf.square(loss.d_fake-b)]
        elif self.config.fitness_method == "least_squares5":
            c = gan.loss.config.labels[2]
            b = gan.loss.config.labels[1]
            a = gan.loss.config.labels[0]
            self.gang_loss = tf.nn.relu(loss.d_real)*loss.d_fake
        elif self.config.fitness_method == "least_squares7":
            c = gan.loss.config.labels[2]
            b = gan.loss.config.labels[1]
            a = gan.loss.config.labels[0]
            self.gang_loss = (tf.nn.relu(loss.d_real)*tf.nn.relu(loss.d_fake))
        elif self.config.fitness_method == "least_squares8":
            c = gan.loss.config.labels[2]
            b = gan.loss.config.labels[1]
            a = gan.loss.config.labels[0]
            self.gang_loss = (loss.d_real-a)*(loss.d_fake-a)
        elif self.config.fitness_method == "least_squares9":
            c = gan.loss.config.labels[2]
            b = gan.loss.config.labels[1]
            a = gan.loss.config.labels[0]
            self.gang_loss = (loss.d_real-a)*(loss.d_fake-a)-(loss.d_real-b)*(loss.d_fake-b)
        elif self.config.fitness_method == "least_squares6":
            c = gan.loss.config.labels[2]
            b = gan.loss.config.labels[1]
            a = gan.loss.config.labels[0]
            self.gang_loss = tf.square(tf.nn.relu(loss.d_real))*tf.sign(loss.d_fake)*tf.square(loss.d_fake)

        elif self.config.fitness_method == "raleast_squares":
            c = gan.loss.config.labels[2]
            b = gan.loss.config.labels[1]
            a = gan.loss.config.labels[0]
            self.gang_loss = tf.square((-(loss.d_fake - loss.d_real)+1)/2) - tf.square((-(loss.d_real-loss.d_fake)+1)/2)
        elif self.config.fitness_method == "jsd":
            def mean_entropy(p, q):
                return p * (tf.log(p / (0.5*(p+q))))
            p = loss.d_fake
            q = loss.d_real
            self.gang_loss = -((mean_entropy(q, p) + mean_entropy(p, q)) / 2)

        elif self.config.fitness_method == "g_loss":
            self.gang_loss = g_loss
        elif self.config.fitness_method == "d_loss":
            self.gang_loss = d_loss
        elif self.config.fitness_method == "-g_loss":
            self.gang_loss = -g_loss
        elif self.config.fitness_method == 'ralsgan':
            b = gan.loss.config.labels[1]
            a = gan.loss.config.labels[0]
            self.gang_loss = 0.5*tf.square(loss.d_real - loss.d_fake - b) + 0.5*tf.square(loss.d_fake - loss.d_real - b)
        elif self.config.fitness_method == 'ragan':
            self.gang_loss = -tf.nn.sigmoid(loss.d_real-loss.d_fake)+ \
                         tf.nn.sigmoid(loss.d_fake-loss.d_real)
        elif self.config.fitness_method == 'ragan2':
            self.gang_loss = tf.log(tf.nn.sigmoid(loss.d_real)+TINY) - \
                         tf.log(tf.nn.sigmoid(loss.d_fake)+TINY)
        elif self.config.fitness_method == 'f-r':
            #self.gang_loss = tf.nn.sigmoid(loss.d_fake - loss.d_real)
            self.gang_loss = 2*tf.nn.sigmoid(2*(loss.d_fake-loss.d_real))-1
        elif self.config.fitness_method == 'f-r2':
            #self.gang_loss = tf.nn.sigmoid(loss.d_fake - loss.d_real)
            self.gang_loss = tf.nn.sigmoid((loss.d_fake))-tf.nn.sigmoid((loss.d_real))
        elif self.config.fitness_method == 'f-r3':
            #self.gang_loss = tf.nn.sigmoid(loss.d_fake - loss.d_real)
            self.gang_loss = tf.nn.sigmoid(loss.d_fake-loss.d_real)-tf.nn.sigmoid(loss.d_real-loss.d_fake)
        else:
            self.gang_loss = loss.d_fake - loss.d_real



        g_vars = list(g_vars)
        d_vars = list(d_vars)
        self.all_g_vars = g_vars
        self.all_d_vars = d_vars

        def random_like(x):
            shape = self.ops.shape(x)
            return tf.random_uniform(shape, minval=-0.01, maxval=0.01)

        self.ug = None#gan.session.run(g_vars)
        self.ud = None#gan.session.run(d_vars)
        self.pg = [tf.zeros_like(v) for v in g_vars]
        self._assign_g = [v.assign(pv) for v,pv in zip(g_vars, self.pg)]
        self.pd = [tf.zeros_like(v) for v in d_vars]
        self._assign_d = [v.assign(pv) for v,pv in zip(d_vars, self.pd)]
        self.pm = tf.zeros([1])
        self.assign_add_d = [v.assign(pv*self.pm+v) for v,pv in zip(d_vars, self.pd)]
        self.assign_add_g = [v.assign(pv*self.pm+v) for v,pv in zip(g_vars, self.pg)]
        self.mutate_g = [v.assign(random_like(v)+v) for v in d_vars]
        self.mutate_d = [v.assign(random_like(v)+v) for v in g_vars]

        self.sgs = []
        self.sds = []

        self.last_fitness_step = 0

    def required(self):
        return ""


    def rank_gs(self, gs):
        # todo fitness?
        return list(np.flip(gs, axis=0)) # most recent

    def rank_ds(self, ds):
        # todo fitness?
        return list(np.flip(ds, axis=0)) # most recent

    def destructive_mixture_g(self, priority_g):
        g_vars = self.all_g_vars
        self.gan.session.run(self._assign_g, {}) # zero
        for i, s in enumerate(self.sgs):
            self.add_g(priority_g[i], s)
        return self.gan.session.run(g_vars)

    def destructive_mixture_d(self, priority_d):
        d_vars = self.all_d_vars
        self.gan.session.run(self._assign_d, {}) # zero
        for i, s in enumerate(self.sds):
            self.add_d(priority_d[i], s)
        return self.gan.session.run(d_vars)

    def nash_memory(self, sg, sd, ug, ud):
        is_sd_nan = np.isnan(np.sum(np.sum(v) for v in sg))
        is_sg_nan = np.isnan(np.sum(np.sum(v) for v in sd))

        if is_sd_nan or is_sg_nan:
            print("NAN detected, falling back to best candidate")
            return [self.sgs[0], self.sds[0]]
        
        #zs = [ self.gan.session.run(self.gan.fitness_inputs()) for i in range(self.config.fitness_test_points or 10)]
        #xs = [ self.gan.session.run(self.gan.inputs.inputs()) for i in range(self.config.fitness_test_points or 10)]
        #self.xs = xs
        #self.zs = zs
        xs = []
        zs = []
        self.sgs = [sg] + self.sgs
        self.sds = [sd] + self.sds

        if isinstance(self.gang_loss, list):
            a = self.payoff_matrix(self.sgs, self.sds, xs, zs, self.gang_loss[0])
            b = self.payoff_matrix(self.sgs, self.sds, xs, zs, self.gang_loss[1])
            print("Payoffa:", a)
            print("Payoffb:", b)
        else:
            print("Calculating nash")
            a = self.payoff_matrix(self.sgs, self.sds, xs, zs)
            if np.min(a) == np.max(a) or np.isnan(np.sum(a)):
                print("WARNING: Degenerate game, skipping")
                print(a)
                self.priority_ds = list(np.zeros(len(self.sds)))
                self.priority_gs = list(np.zeros(len(self.sgs)))
                return [ug, ud]
            print("Payoff:", a)
            b = -a

        is_a_nan = np.isnan(np.sum(np.sum(v) for v in a))
        is_b_nan = np.isnan(np.sum(np.sum(v) for v in b))

        if is_a_nan or is_b_nan:
            print("NAN detected in payoff matrix, falling back to best candidate")
            self.sgs = self.sgs[1:]
            self.sds = self.sds[1:]
            return [self.sgs[0], self.sds[0]]
 
        if self.config.use_nash:
            priority_g, new_ug, priority_d, new_ud = self.nash_mixture_from_payoff(a, b, self.sgs, self.sds)
        elif self.config.use_crossover:
            priority_g = self.mixture_from_payoff(a, 1, self.sgs)
            priority_d = self.mixture_from_payoff(b, 0, self.sds)
        else:
            priority_g = self.mixture_from_payoff(a, 1, self.sgs)
            new_ug = self.destructive_mixture_g(priority_g)

            priority_d = self.mixture_from_payoff(b, 0, self.sds)
            new_ud = self.destructive_mixture_d(priority_d)

        memory_size = self.config.nash_memory_size or 10
        sorted_sgs = [[p, v] for p,v in zip(priority_g, self.sgs)]
        sorted_sds = [[p, v] for p,v in zip(priority_d, self.sds)]
        sorted_sgs.sort(key=lambda x: -x[0])
        sorted_sds.sort(key=lambda x: -x[0])
        print('mixture g:', [x[0] for x in sorted_sgs])
        print('mixture d:', [x[0] for x in sorted_sds])
        self.priority_gs = [x[0] for x in sorted_sgs]
        self.priority_ds = [x[0] for x in sorted_sds]
        sorted_sds = [s[1] for s in sorted_sds]
        sorted_sgs = [s[1] for s in sorted_sgs]
        self.sgs = sorted_sgs[:memory_size]
        self.sds = sorted_sds[:memory_size]
        self.priority_gs = self.priority_gs[:memory_size]
        self.priority_ds = self.priority_ds[:memory_size]

        if self.config.use_crossover:
            new_ug = self.crossover(self.sgs[0],self.sgs[1])
            new_ud = self.crossover(self.sds[0],self.sds[1])


        return [new_ug, new_ud]

    def crossover(self, s1, s2):
        strat = list(s1)
        for i,layer in enumerate(s2):
            mask = np.random.rand(*layer.shape)
            if self.config.crossover_random == None:
                mask = np.around(mask)
            strat[i]=mask*layer + (1-mask)*strat[i]
        return strat

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    def sumdiv(self, x):
        e_x = x
        return e_x / e_x.sum(axis=0)

    def nash_mixture_from_payoff(self, payoffa, payoffb, sgs, sds):
        config = self.config
        def _update_g(p):
            p = np.reshape(p, [-1])
            result = self.destructive_mixture_g(p)
            return p, result

        def _update_d(p):
            p = np.reshape(p, [-1])
            result = self.destructive_mixture_d(p)
            return p, result

        if self.config.nash_method == 'support':
            try:
                u = next(nash.Game(payoffa, payoffb).support_enumeration())
            except(StopIteration):
                print("Nashpy 'support' iteration failed.  Using 1,0,0...")
                u = [list(np.zeros(len(self.sds))), list(np.zeros(len(self.sgs)))]
                u[0][0]=1.
                u[1][0]=1.

        elif self.config.nash_method == 'lemke':
            u = next(nash.Game(payoffa, payoffb).lemke_howson_enumeration())

        else:
            try:
                u = next(nash.Game(payoffa, payoffb).vertex_enumeration())
            except(StopIteration, scipy.spatial.qhull.QhullError):
                print("Nashpy 'vertex' iteration failed.  Using 1,0,0...")
                u = [list(np.zeros(len(self.sds))), list(np.zeros(len(self.sgs)))]
                u[0][0]=1.
                u[1][0]=1.

        if len(u[0]) != len(self.sgs):
            return [None,None,None,None]
        p1, p1result = _update_g(u[0])
        p2, p2result = _update_d(u[1])

        return p1, p1result, p2, p2result


    def mixture_from_payoff(self, payoff, sum_dim, memory):
        u = np.sum(payoff, axis=sum_dim)
        u = self.softmax(u)
        u = np.reshape(u, [len(memory)])
        return u

    def payoff_matrix(self, sgs, sds, xs, zs, method=None):
        self._payoff_matrix = np.zeros([len(sgs), len(sds)])
        result = self._payoff_matrix
        for i, sg in enumerate(sgs):
            for j, sd in enumerate(sds):
                result[i, j]=self.fitness_score(sg, sd, xs, zs, method) # todo fitness ?
        return result

    def fitness_score(self, g, d, xs, zs, method=None):
        self.assign_gd(g,d)
        sum_fitness = 0
        test_points = self.config.fitness_test_points or 10
        if method == None:
            method = self.gang_loss
        for i in range(test_points):
            df, dr = self.gan.session.run([self.gan.loss.d_fake, self.gan.loss.d_real])
            fitness = self.gan.session.run(method)
            sum_fitness += np.average(fitness)
        #for x, z in zip(xs, zs):
        #    loss = self.loss or self.gan.loss
        #    feed_dict = {}
        #    for v, t in zip(x, self.gan.inputs.inputs()):
        #        feed_dict[t]=v
        #    for v, t in zip(z, self.gan.fitness_inputs()):
        #        feed_dict[t]=v
        #    fitness = self.gan.session.run([method], feed_dict)
        #    sum_fitness += np.average(fitness)

        sum_fitness /= float(test_points)
        return sum_fitness

    def assign_gd(self, g, d):
        self.assign_g(g)
        self.assign_d(d)

    def assign_g(self, g):
        fg = {}
        for v, t in zip(g, self.pg):
            fg[t] = v
        self.gan.session.run(self._assign_g, fg)

    def assign_d(self, d):
        fd = {}
        for v, t in zip(d, self.pd):
            fd[t] = v
        self.gan.session.run(self._assign_d, fd)

    def add_g(self, pm, g):
        fg = {}
        for v, t in zip(g, self.pg):
            fg[t] = v
        fg[self.pm] = np.reshape(pm, [1])
        self.gan.session.run(self.assign_add_g, fg)

    def add_d(self, pm, d):
        fd = {}
        for v, t in zip(d, self.pd):
            fd[t] = v
        fd[self.pm] = np.reshape(pm, [1])

        self.gan.session.run(self.assign_add_d, fd)

    def train_g_on_sds(self):
        gan = self.gan
        cd = gan.session.run(self._delegate.d_vars)
        gl = np.zeros(self._delegate.g_loss.shape)
        dl = np.zeros(self._delegate.d_loss.shape)
        for i,sd in enumerate(self.sds):
            p= self.priority_ds[i]
            if(p == 0):
                next
            self.assign_d(sd)

            _gl, _dl, *zs = gan.session.run([self._delegate.g_loss, self._delegate.d_loss]+gan.fitness_inputs())
            print("Train strategy", i, "P", p, "GL", _gl, "DL", _dl)
            gl += _gl * p
            dl += _dl * p
        feed_dict = {}
        for v, t in ([[gl*p, self._delegate.g_loss],[dl*p, self._delegate.d_loss]] + [ [v, t] for v, t in zip(zs, gan.fitness_inputs())]):
            feed_dict[t]=v
        _ = gan.session.run([self._delegate.g_optimizer], feed_dict)
        self.assign_d(cd)

    def train_d_on_sgs(self):
        gan = self.gan
        cg = gan.session.run(self._delegate.g_vars)
        for i,sg in enumerate(self.sgs):
            p= self.priority_gs[i]
            if(p == 0):
                next
            self.assign_g(sg)

            _, _gl, _dl, *zs = gan.session.run([self._delegate.d_optimizer, self._delegate.g_loss, self._delegate.d_loss]+gan.fitness_inputs())
            print("Train strategy", i, "P", p, "GL", _gl, "DL", _dl)

        self.assign_g(cg)

  
    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = gan.loss
        metrics = loss.metrics
        d_vars = self.all_d_vars
        g_vars = self.all_g_vars
        
        if self.ug == None:
            self.ug = gan.session.run(g_vars)
            self.ud = gan.session.run(d_vars)
            self.sgs.append(self.ug)
            self.sds.append(self.ud)
            self.priority_gs = [1]
            self.priority_ds = [1]
            if self.config.preload:
                for preload in self.config.preload:
                    print("Preloading ", preload)
                    default_save_path = os.path.abspath("saves/"+preload)
                    save_file = default_save_path + "/model.ckpt"
                    gan.session.run(tf.global_variables_initializer())
                    if not self.gan.load(save_file):
                        sys.exit("Could not load " + save_file)
                    print("Assigning to sds")
                    preload_ug = gan.session.run(g_vars)
                    preload_ud = gan.session.run(d_vars)
                    self.sgs.append(preload_ug)
                    self.sds.append(preload_ud)
                    self.priority_gs += [0]
                    self.priority_ds += [0]
 
                self.assign_gd(self.ug, self.ud)

        self._delegate.step(feed_dict)

        if config.train_g_on_sds and ((self._delegate.current_step+1) % (config.sds_steps or 100) == 0) and np.max(self.priority_ds) != 0:
            self.train_g_on_sds()
        if config.train_d_on_sgs and ((self._delegate.current_step+1) % (config.sgs_steps or 100) == 0) and np.max(self.priority_gs) != 0:
            self.train_d_on_sgs()
        
        #if self.last_fitness_step == self._delegate.current_step:
        #    return
        self.last_fitness_step=self._delegate.current_step
        #print("Step", self._delegate.current_step+1)
        if (gan.step_count+1) % (config.mix_steps or 100) == 0:
            sg = gan.session.run(g_vars)
            sd = gan.session.run(d_vars)
            if config.nash_memory:
                print("ENAMBLING NASH MEM", len(self.sgs))
                ug, ud = self.nash_memory(sg, sd, self.ug, self.ud)
                print("/ENAMBLING NASH MEM", len(self.sgs))
            else:
                decay = config.decay or 0.5
                ug = [ (o*decay + n*(1-decay)) for o, n in zip(sg, self.ug) ]
                ud = [ (o*decay + n*(1-decay)) for o, n in zip(sd, self.ud) ]

            if config.recreate:
                gan.train_coordinator.request_stop()
                gan.train_coordinator.join(gan.input_threads)
                gan.session.close()
                tf.reset_default_graph()
                #x = tf.contrib.graph_editor.copy(self.gan.inputs.x)
                #self.gan.inputs.x = x
                inputs = hg.inputs.image_loader.ImageLoader(gan.args.batch_size)
                inputs.create(gan.args.directory,
                      channels=gan.x_channels, 
                      format=gan.args.format,
                      crop=gan.args.crop,
                      width=gan.x_width,
                      height=gan.x_height,
                      resize=gan.args.resize)
                config_name = random.choice(self.config.mutations)

                newconfig_file = hg.Configuration.find(config_name+'.json')
                newconfig = hc.Selector().load(newconfig_file)

                newgan = self.gan.config['class'](config=newconfig, inputs=inputs)
                newgan.args = gan.args
                newgan.x_width = gan.x_width
                newgan.x_height = gan.x_height
                newgan.x_channels = gan.x_channels
                newgan.cli = self.gan.cli
                newgan.cli.sampler = None
                gan.cli.sampler = None
                newgan.trainer.sds = self.sds
                newgan.trainer.sgs = self.sgs
                newgan.train_coordinator = tf.train.Coordinator()
                self.sds = None
                self.sgs = None
                self.ug = None
                self.ud = None
                for c in gan.components:
                    c.gan = None
                
                def out(o):
                    return str(type(o))
                self.gan=None
                gc.collect()
                #for v in gc.get_referrers(gan):
                #    try:
                #        print(v.f_locals['self'].__class__) 
                #    except AttributeError:
                #        print("!", v) 
                #print(sys.getrefcount(gan), len(gc.get_referrers(gan)))

                gan.destroy=True
                gan.newgan=newgan
                gan=None
                gc.collect()
                newgan.input_threads = tf.train.start_queue_runners(sess=newgan.session, coord=newgan.train_coordinator)
                newgan.trainer.assign_gd(ug, ud)
                return

            self.assign_gd(ug, ud)

            if self.config.mutate_child and np.random.rand() > (self.config.mutation_chance or 0.7):
                print("Mutating child")
                self.gan.session.run([self.mutate_d, self.mutate_g])

            self.ug = gan.session.run(self.all_g_vars)
            self.ud = gan.session.run(self.all_d_vars)
            if self.current_step < (config.reset_before_step or 0):
                gan.session.run(tf.global_variables_initializer())







