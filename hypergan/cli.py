"""
The command line interface.  Trains a directory of data.
"""
from .configuration import Configuration
from .inputs import *
from .viewer import GlobalViewer
from hypergan.gan_component import ValidationException
from hypergan.gan_component import ValidationException, GANComponent
from time import sleep
import gc
import hyperchamber as hc
import hypergan as hg
import numpy as np
import os
import os
import shutil
import sys
import sys
import tempfile
import time
class CLI:
    def __init__(self, args={}, input_config=None, gan_config=None):
        self.samples = 0
        self.gan_config = gan_config
        self.input_config = input_config

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

        if(self.gan.steps % self.sample_every == 0):
            sample_list = self.sample()

    def create_path(self, filename):
        return os.makedirs(os.path.expanduser(os.path.dirname(filename)), exist_ok=True)

    def create_input(self, blank=False):
        klass = GANComponent.lookup_function(None, self.input_config['class'])
        self.input_config["blank"]=blank
        return klass(self.input_config)

    def build(self):
        return self.gan.build()

    def serve(self, gan):
        return gan_server(self.gan.session, config)

    def sample_forever(self):
        self.gan.inputs.next()
        steps = 0
        while not self.gan.destroy and (steps <= self.args.steps or self.args.steps == -1):
            self.sample()
            steps += 1

    def train(self):
        i=0
        if(self.args.ipython):
            import fcntl
            fd = sys.stdin.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        self.gan = hg.GAN(config=self.gan_config, inputs=self.create_input())
        self.gan.cli = self #TODO remove this link
        self.gan.inputs.next()

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

    def run(self):
        if self.method == 'train':
            self.train()
        elif self.method == 'build':
            self.gan = hg.GAN(config=self.gan_config, inputs=self.create_input(blank=True))
            if not self.gan.load(self.save_file):
                raise ValidationException("Could not load model: "+ self.save_file)
            else:
                if os.path.isfile(self.advSavePath+'advSave.txt'):
                    with open(self.advSavePath+'advSave.txt', 'r') as the_file:
                        content = [x.strip() for x in the_file]
                        self.samples = int(content[1])
                print("Model loaded")
            self.build()
        elif self.method == 'new':
            self.new()
        elif self.method == 'sample':
            self.gan = hg.GAN(config=self.gan_config, inputs=self.create_input(blank=False))
            if not self.gan.load(self.save_file):
                print("Initializing new model")
            else:
                if os.path.isfile(self.advSavePath+'advSave.txt'):
                    with open(self.advSavePath+'advSave.txt', 'r') as the_file:
                        content = [x.strip() for x in the_file]
                        self.samples = int(content[1])
                print("Model loaded")

            self.sample_forever()
