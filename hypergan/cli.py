import argparse
import os
import hyperchamber as hc
from hypergan.gan_component import ValidationException
from . import GAN
from .loaders import *
from .samplers.viewer import GlobalViewer
import hypergan as hg
import time

import fcntl
import os
import sys

from hypergan.samplers.static_batch_sampler import StaticBatchSampler

class CLI:
    def __init__(self, gan, args=None):
        self.samples = 0
        self.steps = 0
        self.gan = gan
        if args == None:
            self.args = self.get_parser().parse_args()
        else:
            self.args = args
        self.samplers = self.sampler_options()
        self.sampler_name = self.get_sampler_name(args)
        if self.sampler_name in self.samplers:
            self.sampler = self.samplers[self.sampler_name]
        else:
            self.sampler = None
        self.validate()

    def load(self):
        #TODO
        pass

    def get_sampler_name(self, args):
        if 'sampler' in args:
            return args['sampler']
        return 'static_batch'

    def sampler_options(self):
        return {
                'static_batch': StaticBatchSampler
        }
        #if(self.args.sampler == "grid"):
        #    sampler = grid_sampler.sample
        #elif(self.args.sampler == "batch"):
        #    sampler = batch_sampler.sample
        #elif(self.args.sampler == "static_batch"):
        #    sampler = static_batch_sampler.sample
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

        sample_list = sampler(self.gan, sample_file)

        return sample_list


    def validate(self):
        if(self.sampler == None):
            raise ValidationException("No sampler found by the name '"+self.sampler_name+"'")

    def common(self, parser):
        parser.add_argument('config', action='store', type=str, help='The configuration file to load.')
        parser.add_argument('directory', action='store', type=str, help='The location of your data.  Subdirectories are treated as different classes.  You must have at least 1 subdirectory.')
        self.common_flags(parser)

    def common_flags(self, parser):
        parser.add_argument('--size', '-s', type=str, default='64x64x3', help='Size of your data.  For images it is widthxheightxchannels.')
        parser.add_argument('--batch_size', '-b', type=int, default=32, help='Number of samples to include in each batch.  If using batch norm, this needs to be preserved when in server mode')
        parser.add_argument('--device', '-d', type=str, default='/gpu:0', help='In the form "/gpu:0", "/cpu:0", etc.  Always use a GPU (or TPU) to train')
        parser.add_argument('--format', '-f', type=str, default='png', help='jpg or png')
        parser.add_argument('--crop', dest='crop', action='store_true', help='If your images are perfectly sized you can skip cropping.')
        parser.add_argument('--align', dest='align', action='store_true', help='Align classes.')
        parser.add_argument('--use_hc_io', type=bool, default=False, help='Set this to no unless you are feeling experimental.')
        parser.add_argument('--save_every', type=int, default=10000, help='Saves the model every n steps.')
        parser.add_argument('--sample_every', type=int, default=10, help='Saves a sample ever X steps.')
        parser.add_argument('--reset_every', type=int, default=None, help='Resets G every n training steps.')
        parser.add_argument('--max_resets', type=int, default=1, help='Will only reset G graph this many times.')
        parser.add_argument('--sampler', type=str, default='static_batch', help='Select a sampler.  Some choices: static_batch, batch, grid, progressive')
        parser.add_argument('--ipython', type=bool, default=False, help='Enables iPython embedded mode.')
        parser.add_argument('--steps', type=int, default=-1, help='Number of steps to train for.  -1 is unlimited (default)')
        parser.add_argument('--viewer', dest='viewer', action='store_true', help='Displays samples in a window.')

    def get_parser(self):
        parser = argparse.ArgumentParser(description='Train, run, and deploy your GANs.', add_help=True)
        subparsers = parser.add_subparsers(dest='method')
        train_parser = subparsers.add_parser('train')
        build_parser = subparsers.add_parser('build')
        serve_parser = subparsers.add_parser('serve')
        subparsers.required = True
        self.common_flags(parser)
        self.common(train_parser)
        self.common(build_parser)
        self.common(serve_parser)

        return parser

    def step(self):
        d_loss, g_loss = self.gan.step()

        if(self.steps > 1 and (self.steps % self.args.sample_every == 0)):
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
        return gan_server(self.sess, config)

    def train(self):
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

            if(args.save_every > 0 and i % args.save_every == 0):
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

    def run(self):
        args = self.args

        if 'config' in args:
            config = args.config
        else:
            config = 'default'

        if 'crop' in args:
            crop = args.crop
        else:
            crop = None

        width, height, channels = self.get_dimensions()

        #config_filename = os.path.expanduser('~/.hypergan/configs/'+args.config+'.json')
        #self.save_file = os.path.expanduser("~/.hypergan/saves/"+args.config+"/model.ckpt")
        #self.create_path(self.save_file)

        #selector = hg.config.selector(args)
        #print("[hypergan] Welcome.  This is one of ", selector.count_configs(), " possible configurations.")
        #config = selector.random_config()

        #print("[hypergan] Config file", config_filename)
        #config = selector.load_or_create_config(config_filename, config)
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
        #tf.train.start_queue_runners(sess=self.sess)

        self.output_graph_size()

        if args.method == 'train':
            self.train()
        elif args.method == 'build':
            self.build()
        elif args.method == 'new':
            self.new()
        #TODO
        #tf.reset_default_graph()
        #self.sess.close()
