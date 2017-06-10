import argparse
import os
import hyperchamber as hc
import tensorflow as tf
from hypergan.gan_component import ValidationException
from . import GAN
from .inputs import *
from .samplers.viewer import GlobalViewer
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

class CLI:
    def __init__(self, gan, args={}):
        self.samples = 0
        self.steps = 0
        self.gan = gan

        self.args = hc.Config(args)

        crop =  self.args.crop

        self.sample_every = self.args.sample_every or 100

        self.samplers = self.sampler_options()
        self.sampler_name = self.get_sampler_name(self.args)
        if self.sampler_name in self.samplers:
            self.sampler = self.samplers[self.sampler_name](self.gan)
        else:
            self.sampler = None

        self.validate()

    def load(self):
        raise ValidationException('Load not implemented')
        #return self.gan.load()

    def get_sampler_name(self, args):
        if 'sampler' in args:
            return args.sampler
        return 'static_batch'

    def sampler_options(self):
        return {
                'static_batch': StaticBatchSampler,
                'batch': BatchSampler,
                'grid': GridSampler
        }
        #elif(self.args.sampler == "progressive"):
        #    sampler = progressive_enhancement_sampler.sample
        #elif(self.args.sampler == "began"):
        #    sampler = began_sampler.sample
        #elif(self.args.sampler == "aligned_began"):
        #    sampler = aligned_began_sampler.sample
        #else:
        #    raise "Cannot find sampler: '"+self.args.sampler+"'"


    def sample(self, sample_file):
        """ Samples to a file.  Useful for visualizing the learning process.

        Use with:

             ffmpeg -i samples/grid-%06d.png -vcodec libx264 -crf 22 -threads 0 grid1-7.mp4

        to create a video of the learning process.
        """

        if(self.args.viewer):
            GlobalViewer.enable()

        sample_list = self.sampler.sample(sample_file)

        return sample_list


    def validate(self):
        if(self.sampler == None):
            raise ValidationException("No sampler found by the name '"+self.sampler_name+"'")

    def step(self):
        self.gan.step()

        if(self.steps > 1 and (self.steps % self.sample_every == 0)):
            sample_file="samples/%06d.png" % (self.samples)
            self.create_path(sample_file)
            sample_list = self.sample(sample_file)
            if self.args.use_hc_io:
                hc.io.sample(self.config, sample_list)

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

        #TODO
        #saver = tf.train.Saver()
        #saver.save(self.sess, build_file)
        print("Saved generator to ", build_file)

    def serve(self, gan):
        return gan_server(self.gan.session, config)

    def train(self):
        args = self.args
        i=0
        if(self.args.ipython):
            fd = sys.stdin.fileno()
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        reset_count=0
        while(i < self.args.steps or self.args.steps == -1):
            i+=1
            start_time = time.time()
            self.step()

            if (args.save_every != None and
                args.save_every != -1 and
                args.save_every > 0 and
                i % args.save_every == 0):
                print(" |= Saving network")
                self.save()
            if args.ipython:
                self.check_stdin()
            end_time = time.time()

    def save(self):
        #TODO
        #saver = tf.train.Saver()
        #saver.save(self.sess, self.save_file)
        return

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

    def get_dimensions(self):
        args = self.args
        if 'size' in args:
            size = args.size or "32"
            split = [int(j) for j in args.size.split("x")]
            if len(split) == 1:
                return [split[0], split[0], 3]
            if len(split) == 2:
                return [split[0], split[1], 3]
            return [split[0], split[1], split[2]]
        return [32, 32, 3]

    def new(self, path):
        if(os.path.exists(path)):
            raise ValidationException('Path does not exist "'+path+'"')

        print("[hypergan] Creating new project '"+path+"'")
        os.mkdir(path)
        os.mkdir(path+'/samples')
        os.mkdir(path+'/saves')
        template = 'default.json' #TODO
        source_configuration = Configuration.find(template)
        json_path = path + '/' + template
        shutil.copyfile(source_configuration, json_path)

        return

    def run(self):
        width, height, channels = self.get_dimensions()
        self.gan.create()

        #self.save_file = os.path.expanduser("~/.hypergan/saves/"+args.config+"/model.ckpt")
        #self.create_path(self.save_file)

        #selector = hg.config.selector(args)
        #print("[hypergan] Welcome.  This is one of ", selector.count_configs(), " possible configurations.")
        #config = selector.random_config()

        #print("[hypergan] Config file", config_filename)
        #config = hg.config.lookup_functions(config)

        #graph = self.setup_input_graph(
        #        args.format,
        #        args.directory,
        #        args.device,
        #        config,
        #        seconds=None,
        #        bitrate=None,
        #        width=width,
        #        height=height,
        #        channels=channels,
        #        crop=crop
        #)

        #if(int(config['y_dims']) > 1 and not args.align):
        #    print("[discriminator] Class loss is on.  Semi-supervised learning mode activated.")
        #    config['losses'].append(hg.losses.supervised_loss.config())
        #else:
        #    print("[discriminator] Class loss is off.  Unsupervised learning mode activated.")
        #self.config = config
        #self.sess = self.gan.sess

        #samples_path = "~/.hypergan/samples/"+args.config+'/'
        #self.create_path(samples_path)

        #self.gan.load_or_initialize_graph(self.save_file)
        #TODO
        tf.train.start_queue_runners(sess=self.gan.session)

        sample_file="samples/%06d.png" % (self.samples)
        self.sample(sample_file)

        self.output_graph_size()
        if self.args.method == 'train':
            self.train()
        elif self.args.method == 'build':
            self.build()
        elif self.args.method == 'new':
            self.new()
        #TODO
        tf.reset_default_graph()
        self.gan.session.close()
