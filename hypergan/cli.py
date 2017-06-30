import os
import hyperchamber as hc
import tensorflow as tf
from hypergan.gan_component import ValidationException
from . import GAN
from .inputs import *
from .viewer import GlobalViewer
from .configuration import Configuration
import hypergan as hg
import time

import fcntl
import os
import shutil
import sys

from hypergan.samplers.static_batch_sampler import StaticBatchSampler
from hypergan.samplers.batch_sampler import BatchSampler
from hypergan.samplers.grid_sampler import GridSampler
from hypergan.samplers.began_sampler import BeganSampler
from hypergan.samplers.aligned_sampler import AlignedSampler
from hypergan.samplers.autoencode_sampler import AutoencodeSampler
from hypergan.samplers.random_walk_sampler import RandomWalkSampler

from hypergan.losses.supervised_loss import SupervisedLoss
from hypergan.multi_component import MultiComponent
from time import sleep

class CLI:
    def __init__(self, gan, args={}):
        self.samples = 0
        self.steps = 0
        self.gan = gan

        args = hc.Config(args)
        self.args = args

        crop =  self.args.crop

        self.config_name = self.args.config or 'default'
        self.method = args.method or 'test'
        self.total_steps = args.steps or -1
        self.sample_every = self.args.sample_every or 100

        self.sampler = CLI.sampler_for(args.sampler or 'static_batch')(self.gan)

        self.validate()
        if self.args.save_file:
            self.save_file = self.args.save_file
        else:
            default_save_path = os.path.abspath("saves/"+args.config)
            self.save_file = default_save_path + "/model.ckpt"
            self.create_path(self.save_file)

    def sampler_for(name):
        samplers = {
                'static_batch': StaticBatchSampler,
                'random_walk': RandomWalkSampler,
                'batch': BatchSampler,
                'grid': GridSampler,
                'began': BeganSampler,
                'autoencode': AutoencodeSampler,
                'aligned': AlignedSampler
        }
        if name in samplers:
            return samplers[name]
        else:
            print("[hypergan] No sampler found for ", name)
            return name

    def sample(self, sample_file):
        """ Samples to a file.  Useful for visualizing the learning process.

        Use with:

             ffmpeg -i samples/grid-%06d.png -vcodec libx264 -crf 22 -threads 0 grid1-7.mp4

        to create a video of the learning process.
        """

        if(self.args.viewer):
            GlobalViewer.enable()
            config_name = self.config_name
            title = "[hypergan] " + config_name
            GlobalViewer.window.set_title(title)

        sample_list = self.sampler.sample(sample_file, self.args.save_samples)

        return sample_list


    def validate(self):
        if(self.sampler == None):
            raise ValidationException("No sampler found by the name '"+self.sampler_name+"'")

    def step(self):
        self.gan.step()

        if(self.steps % self.sample_every == 0):
            sample_file="samples/%06d.png" % (self.samples)
            self.create_path(sample_file)
            sample_list = self.sample(sample_file)
            if self.args.use_hc_io:
                self.gan.config['model'] = self.args.config
                hc.io.sample(self.gan.config, sample_list)

            self.samples += 1

        self.steps+=1

    def output_graph_size(self):
        def mul(s):
            x = 1
            for y in s:
                x*=y
            return x
        def get_size(v):
            shape = [int(x) for x in v.get_shape()]
            size = mul(shape)
            return [v.name, size/1024./1024.]

        #TODO
        #sizes = [get_size(i) for i in tf.global_variables()]
        #sizes = sorted(sizes, key=lambda s: s[1])
        #print("[hypergan] Top 5 largest variables:", sizes[-5:])
        #size = sum([s[1] for s in sizes])
        #print("[hypergan] Size of all variables:", size)

    def create_path(self, filename):
        return os.makedirs(os.path.expanduser(os.path.dirname(filename)), exist_ok=True)

    def build(self, args):
        build_file = os.path.expanduser("~/.hypergan/builds/"+args.config+"/generator.ckpt")
        self.create_path(build_file)

        print("Saved generator to ", build_file)

    def serve(self, gan):
        return gan_server(self.gan.session, config)

    def sample_forever(self):
        while True:
            sample_file="samples/%06d.png" % (self.samples)
            self.create_path(sample_file)
            self.sample(sample_file)
            self.samples += 1
            print("Sample", self.samples)
            sleep(0.2)


    def train(self):
        i=0
        if(self.args.ipython):
            fd = sys.stdin.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        reset_count=0
        while(i < self.total_steps or self.total_steps == -1):
            i+=1
            start_time = time.time()
            self.step()

            if (self.args.save_every != None and
                self.args.save_every != -1 and
                self.args.save_every > 0 and
                i % self.args.save_every == 0):
                print(" |= Saving network")
                self.gan.save(self.save_file)
            if self.args.ipython:
                self.check_stdin()
            end_time = time.time()

    def check_stdin(self):
        try:
            input = sys.stdin.read()
            if input[0]=="y":
                return
            print("INPUT", input)
            from IPython import embed
            # Misc code
            embed()

        except:
            return

    def new(self, path):
        if(os.path.exists(path)):
            raise ValidationException('Path does not exist "'+path+'"')

        print("[hypergan] Creating new project '"+path+"'")
        os.mkdir(path)
        os.mkdir(path+'/samples')
        template = 'default.json' #TODO
        source_configuration = Configuration.find(template)
        json_path = path + '/' + template
        shutil.copyfile(source_configuration, json_path)

        return

    def add_supervised_loss(self):
        number_classes = self.gan.ops.shape(self.gan.inputs.y)[1]
        if(number_classes > 1):
            print("[discriminator] Class loss is on.  Semi-supervised learning mode activated.")
            print("SELFGAN", self.gan.loss)
            supervised_loss = SupervisedLoss(self.gan, self.gan.config.loss)
            self.gan.loss = MultiComponent(components=[supervised_loss, self.gan.loss], combine='add')
            supervised_loss.create()
            #EWW
        else:
            print("[discriminator] Class loss is off.  Unsupervised learning mode activated.")

    def run(self):
        self.output_graph_size()
        if self.method == 'train':
            self.gan.create()
            self.add_supervised_loss(gan)
            self.gan.session.run(tf.global_variables_initializer())

            tf.train.start_queue_runners(sess=self.gan.session)
            self.train()
            tf.reset_default_graph()
            self.gan.session.close()
        elif self.method == 'build':
            self.gan.create()
n           self.build()
            tf.reset_default_graph()
            self.gan.session.close()
        elif self.method == 'new':
            self.new()
        elif self.method == 'sample':
            self.gan.create()
            if(number_classes > 1):
                if not self.args.noclassloss:
                    print("[discriminator] Class loss is on.  Semi-supervised learning mode activated.")
                    print("SELFGAN", self.gan.loss)
                    supervised_loss = SupervisedLoss(self.gan, self.gan.config.loss)
                    self.gan.loss = MultiComponent(components=[supervised_loss, self.gan.loss], combine='add')
                    supervised_loss.create()
                    self.gan.session.run(tf.global_variables_initializer())
                    #EWW
                else:
                    print("Skipping class loss")
            else:
                print("[discriminator] Class loss is off.  Unsupervised learning mode activated.")

            if not self.gan.load(self.save_file):
                print("Initializing new model")
            else:
                print("Model loaded")

            tf.train.start_queue_runners(sess=self.gan.session)
            self.sample_forever()
            tf.reset_default_graph()
            self.gan.session.close()
