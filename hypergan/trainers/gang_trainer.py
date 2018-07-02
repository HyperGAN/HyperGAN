import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect
import nashpy as nash

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class GangTrainer(BaseTrainer):
    def create(self):
        config = self.config
        gan = self.gan
        d_vars = self.d_vars or gan.discriminator.variables()
        g_vars = self.g_vars or (gan.encoder.variables() + gan.generator.variables())

        self._delegate = self.gan.create_component(config.rbbr, d_vars=d_vars, g_vars=g_vars, loss=self.loss)
        g_vars = list(g_vars) + self._delegate.slot_vars_g
        d_vars = list(d_vars) +  self._delegate.slot_vars_d
        self.all_g_vars = g_vars
        self.all_d_vars = d_vars

        self.ug = None#gan.session.run(g_vars)
        self.ud = None#gan.session.run(d_vars)
        self.pg = [tf.zeros_like(v) for v in g_vars]
        self._assign_g = [v.assign(pv) for v,pv in zip(g_vars, self.pg)]
        self.pd = [tf.zeros_like(v) for v in d_vars]
        self._assign_d = [v.assign(pv) for v,pv in zip(d_vars, self.pd)]
        self.pm = tf.zeros([1])
        self.assign_add_d = [v.assign(pv*self.pm+v) for v,pv in zip(d_vars, self.pd)]
        self.assign_add_g = [v.assign(pv*self.pm+v) for v,pv in zip(g_vars, self.pg)]

        self.sgs = []
        self.sds = []

        self.last_fitness_step = 0

        return self._create()


    def _create(self):
        return self._delegate._create()

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
        should_include_sg = np.isnan(np.sum(np.sum(v) for v in sg)) == False
        should_include_sd = np.isnan(np.sum(np.sum(v) for v in sd)) == False
        zs = [ self.gan.session.run(self.gan.fitness_inputs()) for i in range(self.config.fitness_test_points or 10)]
        xs = [ self.gan.session.run(self.gan.inputs.inputs()) for i in range(self.config.fitness_test_points or 10)]
        if(should_include_sg):
            self.sgs = [sg] + self.sgs
        else:
            print("Skip SG (nan)")
        if(should_include_sd):
            self.sds = [sd] + self.sds
        else:
            print("Skip SD (nan)")

        a = self.payoff_matrix(self.sgs, self.sds, xs, zs)

        if self.config.use_nash:
            priority_g, new_ug, priority_d, new_ud = self.nash_mixture_from_payoff(a, self.sgs, self.sds)
        else:
            priority_g = self.mixture_from_payoff(a, 1, self.sgs)
            new_ug = self.destructive_mixture_g(priority_g)

            priority_d = self.mixture_from_payoff(a, 0, self.sds)
            new_ud = self.destructive_mixture_d(priority_d)

        memory_size = self.config.nash_memory_size or 10
        sorted_sgs = [[p, v] for p,v in zip(priority_g, self.sgs)]
        sorted_sds = [[p, v] for p,v in zip(priority_d, self.sds)]
        sorted_sgs.sort(key=lambda x: -x[0])
        sorted_sds.sort(key=lambda x: -x[0])
        print('mixture g:', [x[0] for x in sorted_sgs])
        print('mixture d:', [x[0] for x in sorted_sds])
        sorted_sds = [s[1] for s in sorted_sds]
        sorted_sgs = [s[1] for s in sorted_sgs]
        self.sgs = sorted_sgs[:memory_size]
        self.sds = sorted_sds[:memory_size]

        return [new_ug, new_ud]

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    def sumdiv(self, x):
        e_x = x
        return e_x / e_x.sum(axis=0)

    def nash_mixture_from_payoff(self, payoff, sgs, sds):
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
                u = next(nash.Game(payoff).support_enumeration())
            except(StopIteration):
                print("Nashpy 'support' iteration failed, trying 'lemke howson'")
                u = next(nash.Game(payoff).lemke_howson_enumeration())

        elif self.config.nash_method == 'lemke':
            u = next(nash.Game(payoff).lemke_howson_enumeration())

        else:
            try:
                u = next(nash.Game(payoff).vertex_enumeration())
            except(StopIteration):
                print("Nashpy 'support' iteration failed, trying 'lemke howson'")
                u = next(nash.Game(payoff).lemke_howson_enumeration())

        if self.config.sign_results == 1:
            p1, p1result = _update_g(u[0])
            p2, p2result = _update_d(u[1])
        else:
            p1, p1result = _update_g(u[1])
            p2, p2result = _update_d(u[0])

        return p1, p1result, p2, p2result


    def mixture_from_payoff(self, payoff, sum_dim, memory):
        u = np.sum(payoff, axis=sum_dim)
        u = self.softmax(u)
        u = np.reshape(u, [len(memory)])
        return u

    def payoff_matrix(self, sgs, sds, xs, zs):
        result = np.zeros([len(sgs), len(sds)])
        for i, sg in enumerate(sgs):
            for j, sd in enumerate(sds):
                result[i, j]=self.fitness_score(sg, sd, xs, zs) # todo fitness ?
        return result

    def fitness_score(self, g, d, xs, zs):
        self.assign_gd(g,d)
        sum_fitness = 0
        for x, z in zip(xs, zs):
            loss = self.loss or self.gan.loss
            feed_dict = {}
            for v, t in zip(x, self.gan.inputs.inputs()):
                feed_dict[t]=v
            for v, t in zip(z, self.gan.fitness_inputs()):
                feed_dict[t]=v
            fitness = self.gan.session.run([self._delegate.d_loss], feed_dict)
            sum_fitness += np.average(fitness)

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
  
    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = self.loss or gan.loss
        metrics = loss.metrics
        d_vars = self.all_d_vars
        g_vars = self.all_g_vars
        
        if self.ug == None:
            self.ug = gan.session.run(g_vars)
            self.ud = gan.session.run(d_vars)
            self.sgs.append(self.ug)
            self.sds.append(self.ud)

        self._delegate.step(feed_dict)
        
        if self.last_fitness_step == self._delegate.current_step:
            return
        self.last_fitness_step=self._delegate.current_step
        if (self._delegate.current_step+1) % (config.mix_steps or 100) == 0:
            sg = gan.session.run(g_vars)
            sd = gan.session.run(d_vars)
            if config.nash_memory:
                ug, ud = self.nash_memory(sg, sd, self.ug, self.ud)
            else:
                decay = config.decay or 0.5
                ug = [ (o*decay + n*(1-decay)) for o, n in zip(sg, self.ug) ]
                ud = [ (o*decay + n*(1-decay)) for o, n in zip(sd, self.ud) ]

            self.assign_gd(ug, ud)
            self.ug = gan.session.run(g_vars)
            self.ud = gan.session.run(d_vars)
            if self.current_step < (config.reset_before_step or 0):
                gan.session.run(tf.global_variables_initializer())







