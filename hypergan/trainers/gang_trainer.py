import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect

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

        return self._create()


    def _create(self):
        return self._delegate._create()

    def required(self):
        return ""

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

        self._delegate.step(feed_dict)

        if (self.current_step+1) % (config.mix_steps or 100) == 0:
            sg = gan.session.run(g_vars)
            sd = gan.session.run(d_vars)
            # TODO Parallel Nash
            def _mixup(o,n):
                if config.mixup:
                    p = np.random.uniform(low=0.0, high=1.0, size=1)
                else:
                    p = 0.5
                return (o*p + n*(1-p))
            ug = [ _mixup(o,n) for o, n in zip(sg, self.ug) ]
            ud = [ _mixup(o,n) for o, n in zip(sd, self.ud) ]
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
            print("GANG step")







