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
        self.ug = None#gan.session.run(g_vars)
        self.ud = None#gan.session.run(d_vars)
        self.pg = [tf.zeros_like(v) for v in g_vars]
        self.assign_g = [v.assign(pv) for v,pv in zip(g_vars, self.pg)]
        self.pd = [tf.zeros_like(v) for v in d_vars]
        self.assign_d = [v.assign(pv) for v,pv in zip(d_vars, self.pd)]

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
    
    def nash_memory(self, sg, sd, ug, ud):
        should_include_sg = True#(self.rank_gs([ug, sg])[0])
        should_include_sd = True#(self.rank_ds([ud, sd])[0])
        zs = [ self.gan.session.run(self.gan.uniform_encoder.sample) for i in range(self.config.fitness_test_points or 10)]
        xs = [ self.gan.session.run(self.gan.inputs.x) for i in range(self.config.fitness_test_points or 10)]
        if(should_include_sg):
            self.sgs = [sg] + self.sgs
        else:
            print("Not updating SGs")
        if(should_include_sd):
            print("Updating SDs, ")
            self.sds = [sd] + self.sds
        else:
            print("Not updating SDs")

        a = self.payoff_matrix(self.sgs, self.sds, xs, zs)

        if self.config.use_nash:
            priority_g, new_ug, priority_d, new_ud = self.nash_mixture_from_payoff(a, self.sgs, self.sds)
        else:
            priority_g, new_ug = self.mixture_from_payoff(a, 1, self.sgs)
            priority_d, new_ud = self.mixture_from_payoff(a, 0, self.sds)

        memory_size = self.config.nash_memory_size or 10
        sorted_sgs = [[p, v] for p,v in zip(priority_g, self.sgs)]
        sorted_sds = [[p, v] for p,v in zip(priority_d, self.sds)]
        sorted_sgs.sort(key=lambda x: x[0])
        sorted_sds.sort(key=lambda x: x[0])
        sorted_sds = [s[1] for s in sorted_sds]
        sorted_sgs = [s[1] for s in sorted_sgs]
        self.sgs = sorted_sgs[:memory_size]
        self.sds = sorted_sds[:memory_size]
        print("/D")
        new_ug_is_better = True#self.rank_gs([ug, new_ug])[0] == new_ug
        new_ud_is_better = True#self.rank_ds([ud, new_ud])[0] == new_ud
        if(new_ug_is_better):
            print("Using new ug")
            ug = new_ug
        else:
            print("Using old ug")

        if(new_ud_is_better):
            print("Using new ud")
            ud = new_ud
        else:
            print("Using old ud")

        return [ug, ud]

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    def sumdiv(self, x):
        e_x = x
        return e_x / e_x.sum(axis=0)

    def nash_mixture_from_payoff(self, payoff, sgs, sds):
        def _update(p, mem):
            p = np.reshape(p, [len(mem)])
            result = [np.zeros_like(m) for m in mem[0]]
            for i, s in enumerate(mem):
                for j, w in enumerate(s):
                    result[j] += p[i] *  w
            return p, result

        if config.nash_method == 'support':
            u = next(nash.Game(payoff).support_enumeration())
        else:
            u = next(nash.Game(payoff).vertex_enumeration())
        if self.config.reverse_results:
            print("u", u[0], u[1])
            p1, p1result = _update(u[0], sgs)
            p2, p2result = _update(u[1], sds)
        else:
            print("u2", u[0], u[1])
            p1, p1result = _update(u[1], sgs)
            p2, p2result = _update(u[0], sds)


        return p1, p1result, p2, p2result


    def mixture_from_payoff(self, payoff, sum_dim, memory):
        u = np.sum(payoff, axis=sum_dim)
        u = self.softmax(u)
        u = np.reshape(u, [len(memory)])
        print(u)
        result = [np.zeros_like(m) for m in memory[0]]
        for i, s in enumerate(memory):
            for j, w in enumerate(s):
                result[j] += u[i] *  w
        return u, result

    def payoff_matrix(self, sgs, sds, xs, zs):
        result = np.zeros([len(sgs), len(sds)])
        for i, sg in enumerate(sgs):
            for j, sd in enumerate(sds):
                result[i, j]=self.fitness_score(sg, sd, xs, zs) # todo fitness ?
        print(result)
        return result

    def fitness_score(self, g, d, xs, zs):
        self.assign_gd(g,d)
        sum_fitness = 0
        for x, z in zip(xs, zs):
            loss = self.loss or self.gan.loss
            #fitness = self.gan.session.run([self._delegate.g_fitness], {self.gan.uniform_encoder.sample: z})
            #fitness = self.gan.session.run([loss.d_fake], {self.gan.uniform_encoder.sample: z})
            fitness = self.gan.session.run([self._delegate.d_loss], {self.gan.uniform_encoder.sample: z, self.gan.inputs.x: x})
            sum_fitness += np.average(fitness)

        return sum_fitness


    def assign_gd(self, g, d):
        fg = {}
        for v, t in zip(g, self.pg):
            fg[t] = v
        self.gan.session.run(self.assign_g, fg)
        fd = {}
        for v, t in zip(d, self.pd):
            fd[t] = v
        self.gan.session.run(self.assign_d, fd)

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = self.loss or gan.loss
        metrics = loss.metrics
        d_vars = self.d_vars or gan.discriminator.variables()
        g_vars = self.g_vars or (gan.encoder.variables() + gan.generator.variables())
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

            fg = {}
            for v, t in zip(ug, self.pg):
                fg[t] = v
            gan.session.run(self.assign_g, fg)
            fd = {}
            for v, t in zip(ud, self.pd):
                fd[t] = v
            gan.session.run(self.assign_d, fd)
            self.ug = gan.session.run(g_vars)
            self.ud = gan.session.run(d_vars)
            if self.current_step < (config.reset_before_step or 0):
                gan.session.run(tf.global_variables_initializer())
                print("RESETTING")
            else:
                print("NOT RESETTING")

            print("GANG step")







