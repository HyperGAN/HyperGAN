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
        self._delegate = self.gan.create_component(self.config.delegate)

    def variables(self):
        return self._delegate.variables()

    def required(self):
        return []

    def step(self, feed_dict):
        gan = self.gan
        sess = gan.session
        config = self.config

        self._delegate.step(feed_dict)

        transition_step = self.curriculum[self.curriculum_index][0]
        self.current_step += 1
        if (self.current_step-1) == transition_step:

            gan.save("saves/curriculum")
            self.curriculum_index+=1

            if self.config.cycle:
                self.curriculum_index = self.curriculum_index % len(self.curriculum)
            if self.curriculum_index == len(self.curriculum):
                print("End of curriculum")
                gan.save("saves/curriculum")
                gan.session.close()
                tf.reset_default_graph()
                sys.exit()


            print("Loading index", self.curriculum_index, self.curriculum, self.curriculum[self.curriculum_index])
            gan.session.close()
            tf.reset_default_graph()

            config_name = self.curriculum[self.curriculum_index][1]

            newconfig_file = hg.Configuration.find(config_name+'.json')
            if newconfig_file is None:
                print("Could not find file ", config_name+".json")
                raise("missing file")
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
            newgan.name=config_name
            newgan.trainer.curriculum= self.curriculum
            newgan.trainer.curriculum_index= self.curriculum_index
            newgan.trainer.config.cycle = self.config.cycle
            newgan.cli.sampler = None
            gan.cli.sampler = None
            gan.destroy=True
            gan.newgan=newgan
            gan=None
            gc.collect()
            newgan.load("saves/curriculum")

