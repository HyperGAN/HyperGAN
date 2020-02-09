"""
The command line interface.  Trains a directory of data.
"""
import gc
import sys
import os
import hyperchamber as hc
import numpy as np
from hypergan.gan_component import ValidationException
from .inputs import *
from .viewer import GlobalViewer
from .configuration import Configuration
import hypergan as hg
import time

import os
import shutil
import sys
import tempfile

from hypergan.losses.supervised_loss import SupervisedLoss
from hypergan.multi_component import MultiComponent
from time import sleep


class CLI:
    def __init__(self, args={}, gan_fn=None, inputs_fn=None, gan_config=None):
        self.samples = 0
        self.steps = 0
        self.gan_fn = gan_fn
        self.gan_config = gan_config
        self.inputs_fn = inputs_fn

        args = hc.Config(args)
        self.args = args

        crop =  self.args.crop

        self.config_name = self.args.config or 'default'
        self.method = args.method or 'test'
        self.total_steps = args.steps or -1
        self.sample_every = self.args.sample_every or 100

        self.sampler_name = args.sampler
        self.sampler = None

        self.loss_every = self.args.loss_every or 1

        if (self.args.save_losses):
            import matplotlib.pyplot as plt
            self.arr = []
            self.fig,self.ax = plt.subplots()
            self.temp = 0

        self.advSavePath = os.path.abspath("saves/"+self.config_name)+"/"
        if self.args.save_file:
            self.save_file = self.args.save_file + "/"
        else:
            default_save_path = os.path.abspath("saves/"+self.config_name)
            self.save_file = default_save_path + "/"
            self.create_path(self.save_file)

        title = "[hypergan] " + self.config_name
        GlobalViewer.set_options(
            enable_menu = self.args.menu,
            title = title,
            viewer_size = self.args.viewer_size,
            enabled = self.args.viewer,
            zoom = self.args.zoom)

    def sample(self, allow_save=True):
        """ Samples to a file.  Useful for visualizing the learning process.

        If allow_save is False then saves will not be created.

        Use with:

             ffmpeg -i samples/grid-%06d.png -vcodec libx264 -crf 22 -threads 0 grid1-7.mp4

        to create a video of the learning process.
        """
        sample_file="samples/%s/%06d.png" % (self.config_name, self.samples)
        self.create_path(sample_file)
        self.lazy_create()
        sample_list = self.sampler.sample(sample_file, allow_save and self.args.save_samples)
        if allow_save:
            self.samples += 1

        return sample_list

    def lazy_create(self):
        if(self.sampler == None):
            self.sampler = self.gan.sampler_for(self.sampler_name)(self.gan, samples_per_row=self.args.width)
            if(self.sampler == None):
                raise ValidationException("No sampler found by the name '"+self.sampler_name+"'")

    def step(self):
        self.gan.step()

        if(self.steps % self.sample_every == 0 and self.args.sampler):
            sample_list = self.sample()

        self.steps+=1

    def create_path(self, filename):
        return os.makedirs(os.path.expanduser(os.path.dirname(filename)), exist_ok=True)

    def build(self):
        return self.gan.build()

    def serve(self, gan):
        return gan_server(self.gan.session, config)

    def sample_forever(self):
        while not self.gan.destroy:
            self.sample()

    def train(self):
        i=0
        if(self.args.ipython):
            import fcntl
            fd = sys.stdin.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        self.inputs = self.inputs_fn()
        self.gan = self.gan_fn(self.gan_config, self.inputs)
        self.gan.cli = self #TODO remove this link

        self.gan.initialize_variables()
        if self.gan.load(self.save_file):
            print("Model loaded")
        else:
            print("Initializing new model")

        self.sample()

        while((i < self.total_steps or self.total_steps == -1) and not self.gan.destroy):
            i+=1
            start_time = time.time()
            self.step()

            if (self.args.save_every != None and
                self.args.save_every != -1 and
                self.args.save_every > 0 and
                i % self.args.save_every == 0):
                print(" |= Saving network")
                self.gan.save(self.save_file)   
                self.create_path(self.advSavePath+'advSave.txt')
                if os.path.isfile(self.advSavePath+'advSave.txt'):
                    with open(self.advSavePath+'advSave.txt', 'w') as the_file:
                        the_file.write(str(self.steps)+"\n")
                        the_file.write(str(self.samples)+"\n")
            if self.args.ipython:
                self.check_stdin()
            end_time = time.time()

    def check_stdin(self):
        try:
            input = sys.stdin.read()
            if input[0]=="y":
                return
            from IPython import embed
            # Misc code
            embed()

        except:
            return

    def new(self):
        if self.args.toml:
            config_format = '.toml'
        else:
            config_format = '.json'
        template = self.args.directory + config_format
        print("[hypergan] Creating new configuration file '"+template+"' based off of '"+self.config_name+config_format)
        if os.path.isfile(template):
            raise ValidationException("File exists: " + template)
        source_configuration = Configuration.find(self.config_name+config_format, config_format=config_format)
        shutil.copyfile(source_configuration, template)

        return

    def add_supervised_loss(self):
        if self.args.classloss:
            print("[discriminator] Class loss is on.  Semi-supervised learning mode activated.")
            supervised_loss = SupervisedLoss(self.gan, self.gan.config.loss)
            self.gan.loss = MultiComponent(components=[supervised_loss, self.gan.loss], combine='add')
            #EWW
        else:
            print("[discriminator] Class loss is off.  Unsupervised learning mode activated.")

    def run(self):
        if self.method == 'train':
            self.train()
        elif self.method == 'build':
            self.inputs = self.inputs_fn()
            self.gan = self.gan_fn(self.gan_config, self.inputs)
            if not self.gan.load(self.save_file):
                raise ValidationException("Could not load model: "+ self.save_file)
            else:
                if os.path.isfile(self.advSavePath+'advSave.txt'):
                    with open(self.advSavePath+'advSave.txt', 'r') as the_file:
                        content = [x.strip() for x in the_file]
                        self.steps = int(content[0])
                        self.samples = int(content[1])
                print("Model loaded")
            self.build()
        elif self.method == 'new':
            self.new()
        elif self.method == 'sample':
            self.inputs = self.inputs_fn()
            self.gan = self.gan_fn(self.gan_config, self.inputs)
            self.gan.initialize_variables()
            if not self.gan.load(self.save_file):
                print("Initializing new model")
            else:
                if os.path.isfile(self.advSavePath+'advSave.txt'):
                    with open(self.advSavePath+'advSave.txt', 'r') as the_file:
                        content = [x.strip() for x in the_file]
                        self.steps = int(content[0])
                        self.samples = int(content[1])
                print("Model loaded")

            self.sample_forever()
