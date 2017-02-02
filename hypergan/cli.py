import argparse
import os
import tensorflow as tf
import hyperchamber as hc
from hypergan.util.gan_server import *
from hypergan.util.globals import *
from . import GAN
from .loaders import *
import hypergan as hg
import time

from hypergan.util.ops import *
from hypergan.samplers import *

class CLI:
    def __init__(self):
        self.sampled = 0
        self.steps = 0
        self.run()

    def common(self, parser):
        parser.add_argument('directory', action='store', type=str, help='The location of your data.  Subdirectories are treated as different classes.  You must have at least 1 subdirectory.')
        self.common_flags(parser)

    def common_flags(self, parser):
        parser.add_argument('--size', '-s', type=str, default='64x64x3', help='Size of your data.  For images it is widthxheightxchannels.')
        parser.add_argument('--batch_size', '-b', type=int, default=32, help='Number of samples to include in each batch.  If using batch norm, this needs to be preserved when in server mode')
        parser.add_argument('--config', '-c', type=str, default=None, help='The name of the config.  This is used for loading/saving the model and configuration.')
        parser.add_argument('--device', '-d', type=str, default='/gpu:0', help='In the form "/gpu:0", "/cpu:0", etc.  Always use a GPU (or TPU) to train')
        parser.add_argument('--format', '-f', type=str, default='png', help='jpg or png')
        parser.add_argument('--crop', type=bool, default=False, help='If your images are perfectly sized you can skip cropping.')
        parser.add_argument('--use_hc_io', type=bool, default=False, help='Set this to no unless you are feeling experimental.')
        parser.add_argument('--save_every', type=int, default=10000, help='Saves the model every n steps.')
        parser.add_argument('--sample_every', type=int, default=10, help='Saves a sample ever X steps.')
        parser.add_argument('--sampler', type=str, default='static_batch', help='Select a sampler.  Some choices: static_batch, batch, grid, progressive')

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


    def sample(self, sample_file):
        """ Samples to a file.  Useful for visualizing the learning process.

        Use with:

             ffmpeg -i samples/grid-%06d.png -vcodec libx264 -crf 22 -threads 0 grid1-7.mp4

        to create a video of the learning process.
        """

        if(self.args.sampler == "grid"):
            sampler = grid_sampler.sample
        elif(self.args.sampler == "batch"):
            sampler = batch_sampler.sample
        elif(self.args.sampler == "static_batch"):
            sampler = static_batch_sampler.sample
        elif(self.args.sampler == "progressive"):
            sampler = progressive_enhancement_sampler.sample
        else:
            raise "Cannot find sampler: '"+args.sampler+"'"

        sample_list = sampler(sample_file, self.sess, self.config)

        return sample_list

    def step(self):
        d_loss, g_loss = self.config['trainer.train'](self.sess, self.config)

        if(self.steps > 1 and (self.steps % self.args.sample_every == 0)):
            sample_file="samples/%06d.png" % (self.sampled)
            self.create_path(sample_file)
            print(str(self.steps)+":", "Sample created "+sample_file)
            sample_list = self.sample(sample_file)
            if self.args.use_hc_io:
                hc.io.sample(self.config, sample_list)


            self.sampled += 1

        self.steps+=1
        return True

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

        sizes = [get_size(i) for i in tf.all_variables()]
        sizes = sorted(sizes, key=lambda s: s[1])
        print("[hypergan] Top 5 largest variables:", sizes[-5:])
        size = sum([s[1] for s in sizes])
        print("[hypergan] Size of all variables:", size)

    def create_path(self, filename):
        return os.makedirs(os.path.expanduser(os.path.dirname(filename)), exist_ok=True)

    def build(self, args):
        build_file = os.path.expanduser("~/.hypergan/builds/"+args.config+"/generator.ckpt")
        self.create_path(build_file)

        saver = tf.train.Saver()
        saver.save(self.sess, build_file)
        print("Saved generator to ", build_file)

    def serve(self, gan):
        return gan_server(self.sess, config)

    def train(self, args):
        sampled=False
        i=0
        while(True):
            i+=1
            start_time = time.time()
            with tf.device(args.device):
              self.step()
            if(args.save_every != 0 and i % args.save_every == 0):
                print(" |= Saving network")
                saver = tf.train.Saver()
                saver.save(self.sess, self.save_file)
            end_time = time.time()

    def setup_input_graph(self, format, directory, device, config, seconds=None,
            bitrate=None, crop=False, width=None, height=None, channels=3):
        x,y,f,num_labels,examples_per_epoch=self.setup_input_loader(format, 
                directory, 
                device, 
                config, 
                seconds=seconds,
                bitrate=bitrate, 
                crop=crop, 
                width=width, 
                height=height, 
                channels=channels)
        return {
                'x':x,
                'y':y,
                'f':f,
                'num_labels':num_labels,
                'examples_per_epoch':examples_per_epoch
            }

    def setup_input_loader(self, format, directory, device, config, seconds=None,
            bitrate=None, crop=False, width=None, height=None, channels=3):
        with tf.device('/cpu:0'):
            #TODO mp3 braken
            if(format == 'mp3'):
                return audio_loader.mp3_tensors_from_directory(
                        directory,
                        config['batch_size'],
                        seconds=seconds,
                        channels=channels,
                        bitrate=bitrate,
                        format=format)
            else:
                return image_loader.labelled_image_tensors_from_directory(
                        directory,
                        config['batch_size'], 
                        channels=channels, 
                        format=format,
                        crop=crop,
                        width=width,
                        height=height)

    def run(self):
        parser = self.get_parser()
        self.args = parser.parse_args()
        args = self.args
        if args.config is None:
            parser.error("the following arguments are required: --config")

        crop = args.crop
        width = int(args.size.split("x")[0])
        height = int(args.size.split("x")[1])
        channels = int(args.size.split("x")[2])

        config_filename = os.path.expanduser('~/.hypergan/configs/'+args.config+'.json')
        self.save_file = os.path.expanduser("~/.hypergan/saves/"+args.config+".ckpt")

        selector = hg.config.selector(args)
        print("[hypergan] Welcome.  This is one of ", selector.count_configs(), " possible configurations.")
        config = selector.random_config()

        print("[hypergan] Config file", config_filename)
        config = selector.load_or_create_config(config_filename, config)
        config['dtype']=tf.float32 #TODO fix.  this happens because dtype is stored as an enum
        config['model']=args.config
        config['batch_size'] = args.batch_size
        config = hg.config.lookup_functions(config)

        graph = self.setup_input_graph(
                args.format,
                args.directory,
                args.device,
                config,
                seconds=None,
                bitrate=None,
                width=width,
                height=height,
                channels=channels,
                crop=crop
        )
        config['y_dims']=graph['num_labels']
        config['x_dims']=[height,width]
        config['channels']=channels

        self.config = config
        self.gan = GAN(config, graph, device=args.device)
        self.sess = self.gan.sess

        samples_path = "~/.hypergan/samples/"+args.config+'/'
        self.create_path(samples_path)

        self.gan.load_or_initialize_graph(self.save_file)
        tf.train.start_queue_runners(sess=self.sess)

        self.output_graph_size()

        if args.method == 'train':
            self.train(args)
        elif args.method == 'serve':
            self.serve(args)
        elif args.method == 'build':
            self.build(args)

        tf.reset_default_graph()
        self.sess.close()
