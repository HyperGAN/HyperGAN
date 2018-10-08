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

class CurriculumTrainer(BaseTrainer):
    def create(self):
        self.curriculum = self.config.curriculum
        self.curriculum_index = 0
        self.transition_step = self.curriculum[0][0]
        d_vars = self.d_vars or gan.discriminator.variables()
        g_vars = self.g_vars or (gan.encoder.variables() + gan.generator.variables())
        loss = self.gan.loss
        self._delegate = self.gan.create_component(self.config.delegate, d_vars=d_vars, g_vars=g_vars, loss=loss)

    def required(self):
        return ["curriculum"]

    def _step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config

        self._delegate.step(feed_dict)

        if self.current_step == self.transition_step:

            gan.save("saves/curriculum")
            self.curriculum_index+=1
            self.transition_step = self.curriculum[self.curriculum_index][0]
            gan.train_coordinator.request_stop()
            gan.train_coordinator.join(gan.input_threads)
            gan.session.close()
            tf.reset_default_graph()

            config_name = self.curriculum[self.curriculum_index][1]

            newconfig_file = hg.Configuration.find(config_name+'.json')
            print("=> Loading config file", newconfig_file)
            newconfig = hc.Selector().load(newconfig_file)
            if 'inherit' in newconfig:
                base_filename = hg.Configuration.find(newconfig['inherit']+'.json')
                base_config = hc.Selector().load(base_filename)
                newconfig = hc.Config({**base_config, **newconfig})

            inputs = hg.inputs.image_loader.ImageLoader(newconfig.runtime['batch_size'])
            inputs.create(gan.args.directory,
                  channels=newconfig.runtime['channels'], 
                  format=gan.args.format,
                  crop=gan.args.crop,
                  width=newconfig.runtime['width'],
                  height=newconfig.runtime['height'],
                  resize=gan.args.resize)

            newgan = gan.config['class'](config=newconfig, inputs=inputs)
            newgan.args = gan.args
            newgan.cli = self.gan.cli
            newgan.trainer.curriculum_index= self.curriculum_index
            newgan.trainer.transition_step = self.transition_step
            newgan.cli.sampler = None
            gan.cli.sampler = None
            gan.destroy=True
            gan.newgan=newgan
            gan=None
            gc.collect()
            newgan.load("saves/curriculum")
            newgan.train_coordinator = tf.train.Coordinator()
            newgan.input_threads = tf.train.start_queue_runners(sess=newgan.session, coord=newgan.train_coordinator)

