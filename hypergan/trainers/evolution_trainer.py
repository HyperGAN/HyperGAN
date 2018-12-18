import tensorflow as tf
import numpy as np
import hyperchamber as hc
import inspect

from hypergan.trainers.base_trainer import BaseTrainer

TINY = 1e-12

class EvolutionTrainer(BaseTrainer):
    def _create(self):
        gan = self.gan
        generator = self.gan.generator
        config = self.config

        d_vars = self.d_vars or gan.discriminator.variables()

        loss = self.loss or gan.loss
        d_loss, g_loss = loss.sample

        self.d_log = -tf.log(tf.abs(d_loss+TINY))


        d_optimizer = self.build_optimizer(config, 'd_', config.d_trainer, self.d_lr, d_vars, d_loss)
        #TODO more than one g_loss
        g_optimizer = [self.build_optimizer(config, 'g_', config.g_trainer, self.g_lr, g.variables(), g_loss) for g, l in zip(generator.children, loss.children_losses)]

        assign_children = []
        for p, o in generator.parent_child_tuples:
            for ov, pv in zip(o.variables(), p.variables()):
                op=tf.assign(ov, pv)
                if config.mutation_percent:
                    op += tf.random_normal(self.gan.ops.shape(pv), mean=0, stddev=0.01) * tf.cast(tf.greater(config.mutation_percent, tf.random_uniform(shape=self.gan.ops.shape(pv), minval=0, maxval=1)), tf.float32)
                assign_children.append(op)
        self.clone_parent = tf.group(*assign_children)


        update_parent=[]
        for p, o in generator.parent_child_tuples:
            c_to_p = []
            for ov, pv in zip(o.variables(), p.variables()):
                op=tf.assign(pv, ov)
                c_to_p.append(op)
            update_parent.append(tf.group(*c_to_p))
        self.update_parent = update_parent
        f_lambda = config.f_lambda or 1

        def _squash(grads):
            return tf.add_n([tf.reshape(gan.ops.squash(g), [1]) for g in grads])
        children_grads = [_squash(tf.gradients(l, d_vars)) for l in loss.children_losses]
        if config.fitness == "g":
            self.measure_g = [-l for l in loss.children_losses]
        else:
            self.measure_g = [-l+f_lambda*(-tf.log(TINY+grad_d - tf.log(TINY+tf.nn.sigmoid(loss.d_loss)) - tf.log(TINY+1-tf.nn.sigmoid(l)))) for l, grad_d in zip(loss.children_losses, children_grads)]
        loss.metrics['measure_g'] = tf.reduce_mean(self.measure_g)
        loss.metrics['g_loss'] = loss.g_loss
        loss.metrics['d_loss'] = loss.d_loss

        self.g_loss = g_loss
        self.d_loss = d_loss
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.hist = [0 for i in range(len(self.gan.generator.children))]

        return g_optimizer, d_optimizer

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config
        loss = self.loss or gan.loss
        metrics = loss.metrics
        generator = gan.generator

        d_loss, g_loss = loss.sample

        #winner = np.random.choice(range(len(gan.generator.children)))
        winners = []
        
        for i in range(len(generator.parents)):
            child_count = generator.config.child_count
            choices = self.measure_g[i*child_count:(i+1)*child_count]
            choice = np.argmax(sess.run(choices))
            winner = i*child_count + choice
            self.hist[winner]+=1
            winners.append(winner)
        sess.run([self.update_parent[winner] for winner in winners])
        for i in range(config.d_update_steps or 1):
            sess.run(self.d_optimizer)

        sess.run(self.clone_parent)
        for i in range(config.g_update_steps or 1):
            sess.run(self.g_optimizer)
        measure_g = sess.run(self.measure_g)

        if self.current_step % 100 == 0:
            hist_output = "  " + "".join(["G"+str(i)+":"+str(v)+" "for i, v in enumerate(self.hist)])
            metric_values = sess.run(self.output_variables(metrics), feed_dict)
            print(str(self.output_string(metrics) % tuple([self.current_step] + metric_values)+hist_output))
            self.hist = [0 for i in range(len(self.gan.generator.children))]

