"""
The command line interface.  Trains a directory of data.
"""
from .configuration import Configuration
from .inputs import *
from hypergan.gan_component import ValidationException
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.process_manager import ProcessManager
from hypergan.trainable_gan import TrainableGAN
from time import sleep
import gc
import hyperchamber as hc
import hypergan as hg
import numpy as np
import os
import shutil
import sys
import sys
import tempfile
import time

class CLI:
    def __init__(self, args={}, input_config=None, gan_config=None):
        self.steps = 0
        self.should_sample=False
        self.gan_config = gan_config
        self.input_config = input_config

        args = hc.Config(args)
        self.args = args

        self.devices = args.devices
        crop =  self.args.crop

        self.config_name = self.args.config or 'default'
        self.method = args.method or 'test'
        self.total_steps = args.steps or -1
        self.sample_every = self.args.sample_every or 100

        self.sampler_name = args.sampler
        self.sampler = None
        self.sample_path = "samples/%s" % self.config_name

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

    def lazy_create(self):
        if(self.sampler == None):
            self.sampler = self.gan.sampler_for(self.sampler_name)(self.gan, samples_per_row=self.args.width)
            if(self.sampler == None):
                raise ValidationException("No sampler found by the name '"+self.sampler_name+"'")

    def step(self):
        self.steps+=1
        self.trainable_gan.step()

        if(self.steps % self.sample_every == 0):
            sample_list = self.trainable_gan.sample(self.sampler, self.sample_path)

    def create_path(self, filename):
        return os.makedirs(os.path.expanduser(os.path.dirname(filename)), exist_ok=True)

    def create_input(self, blank=False, rank=None):
        klass = GANComponent.lookup_function(None, self.input_config['class'])
        self.input_config["blank"]=blank
        self.input_config["rank"]=rank
        return klass(self.input_config)

    def build(self):
        return self.gan.build()

    def serve(self, gan):
        return gan_server(self.gan.session, config)


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
        print("Devices D:")
        for component in self.gan.discriminator_components():
            print(component.device)
        print("Devices G:")
        for component in self.gan.generator_components():
            print(component.device)

        if allow_save:
            self.samples += 1

        return sample_list



    def sample_forever(self):
        self.gan.inputs.next()
        self.lazy_create()
        self.trainable_gan = hg.TrainableGAN(self.gan, save_file = self.save_file, devices = self.devices, backend_name = self.args.backend)

        if self.trainable_gan.load():
            print("Model loaded")
        else:
            print("Could not load save")
            return
        steps = 0
        self.gan.cli = self #TODO remove this link
        self.lazy_create()
        while not self.gan.destroy and (steps <= self.args.steps or self.args.steps == -1):
            self.trainable_gan.sample(self.sampler, self.sample_path)
            steps += 1

    def train(self):
        i=0
        if(self.args.ipython):
            import fcntl
            fd = sys.stdin.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        self.gan = hg.GAN(config=self.gan_config, inputs=self.create_input(), device=self.args.parameter_server_device)
        self.gan.cli = self #TODO remove this link
        self.gan.inputs.next()
        self.lazy_create()

        self.trainable_gan = hg.TrainableGAN(self.gan, save_file = self.save_file, devices = self.devices, backend_name = self.args.backend)

        if self.trainable_gan.load():
            print("Model loaded")
        else:
            print("Initializing new model")

        self.trainable_gan.sample(self.sampler, self.sample_path)

        while((self.steps < self.total_steps or self.total_steps == -1) and not self.gan.destroy):
            self.step()
            if self.should_sample:
                self.should_sample = False
                self.sample(False)

            if (self.args.save_every != None and
                self.args.save_every != -1 and
                self.args.save_every > 0 and
                self.steps % self.args.save_every == 0):
                print(" |= Saving network")
                self.trainable_gan.save()
                self.create_path(self.advSavePath+'advSave.txt')
                if os.path.isfile(self.advSavePath+'advSave.txt'):
                    with open(self.advSavePath+'advSave.txt', 'w') as the_file:
                        the_file.write(str(self.samples)+"\n")
            if self.args.ipython:
                self.check_stdin()
        print("Done training model.  Saving")
        self.trainable_gan.save()
        print("============================")
        print("HyperGAN model trained")
        print("============================")

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
        source_configuration = Configuration.find(self.config_name+config_format, config_format=config_format, prepackaged=True)
        shutil.copyfile(source_configuration, template)

        return

    def run(self):
        if self.method == 'train':
            self.train()
        elif self.method == 'build':
            self.gan = hg.GAN(config=self.gan_config, inputs=self.create_input(blank=True))
            if not self.gan.load(self.save_file):
                raise ValidationException("Could not load model: "+ self.save_file)
            self.build()
        elif self.method == 'new':
            self.new()
        elif self.method == 'sample':
            self.gan = hg.GAN(config=self.gan_config, inputs=self.create_input(blank=False))
            if not self.gan.load(self.save_file):
                print("Initializing new model")

            self.sample_forever()

