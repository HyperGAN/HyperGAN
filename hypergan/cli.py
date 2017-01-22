import argparse
import os
import tensorflow as tf
import hyperchamber as hc
from . import GAN
from .loaders import *
import hypergan as hg
import time

class CLI:
    def __init__(self):
        self.sampled = 0
        self.batch_no = 0
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
        parser.add_argument('--epochs', type=int, default=10000, help='The number of iterations through the data before stopping training.')
        parser.add_argument('--save_every', type=int, default=10, help='Saves the model every n epochs.')
        parser.add_argument('--frame_sample', type=str, default=None, help='Frame sampling is used for video creation.')

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


    #TODO fixme
    def frame_sample(self, sample_file, sess, config):
        """ Samples every frame to a file.  Useful for visualizing the learning process.

        Use with:

             ffmpeg -i samples/grid-%06d.png -vcodec libx264 -crf 22 -threads 0 grid1-7.mp4

        to create a video of the learning process.
        """

        if(self.args.frame_sample == None):
            return None
        if(self.args.frame_sample == "grid"):
            frame_sampler = grid_sampler.sample
        else:
            raise "Cannot find frame sampler: '"+args.frame_sample+"'"

        frame_sampler(sample_file, self.sess, config)


    def epoch(self):
        config = self.config
        sess = self.sess
        batch_size = config["batch_size"]
        n_samples =  config['examples_per_epoch']
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            if(i % 10 == 1):
                sample_file="samples/grid-%06d.png" % (self.sampled)
                self.frame_sample(sample_file, sess, config)
                self.sampled += 1


            d_loss, g_loss = config['trainer.train'](sess, config)

        self.batch_no+=1
        return True

    def collect_measurements(self, epoch, sess, config, time):
        d_loss = get_tensor("d_loss")
        d_loss_fake = get_tensor("d_fake_sig")
        d_loss_real = get_tensor("d_real_sig")
        g_loss = get_tensor("g_loss")
        d_class_loss = get_tensor("d_class_loss")
        simple_g_loss = get_tensor("g_loss_sig")

        gl, dl, dlr, dlf, dcl,sgl = sess.run([g_loss, d_loss, d_loss_real, d_loss_fake, d_class_loss, simple_g_loss])
        return {
                "g_loss": gl,
                "g_loss_sig": sgl,
                "d_loss": dl,
                "d_loss_real": dlr,
                "d_loss_fake": dlf,
                "d_class_loss": dcl,
                "g_strength": (1-(dlr))*(1-sgl),
                "seconds": time/1000.0
                }


    #TODO
    def test_epoch(self, epoch, start_time, end_time):
        sample = []
        sample_list = config['sampler'](self.sess,self.config)
        measurements = self.collect_measurements(epoch, self.sess, self.config, end_time - start_time)
        if self.args.use_hc_io:
            hc.io.measure(self.config, measurements)
            hc.io.sample(self.config, sample_list)
        else:
            print("Offline sample created:", sample_list)

    #TODO
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
        build_file = "~/.hypergan/builds/"+args.config+"/generator.ckpt"
        self.create_path(build_file)

        saver = tf.train.Saver()
        saver.save(self.sess, build_file)
        print("Saved generator to ", build_file)

    def serve(self, gan):
        return gan_server(gan.sess, config)

    def train(self, args):
        sampled=False
        print("Running for ", args.epochs, " epochs")
        for i in range(args.epochs):
            start_time = time.time()
            with tf.device(args.device):
                if(not self.epoch()):
                    print("Epoch failed")
                    break
            print("Checking save "+ str(i))
            if(args.save_every != 0 and i % args.save_every == args.save_every-1):
                print(" |= Saving network")
                saver = tf.train.Saver()
                saver.save(self.sess, save_file)
            end_time = time.time()
            self.test_epoch(i, start_time, end_time)

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
        args = parser.parse_args()
        if args.config is None:
            parser.error("the following arguments are required: --config")

        crop = args.crop
        width = int(args.size.split("x")[0])
        height = int(args.size.split("x")[1])
        channels = int(args.size.split("x")[2])


        config_filename = '~/.hypergan/configs/'+args.config+'.json'
        save_file = "~/.hypergan/saves/"+args.config+".ckpt"

        selector = hg.config.selector(args)
        print("[hypergan] Welcome.  You are one of ", selector.count_configs(), " possible configurations.")

        config = selector.random_config()
        config['dtype']=tf.float32 #TODO fix.  this happens because dtype is stored as an enum
        config['batch_size'] = args.batch_size

        config = selector.load_or_create_config(config_filename, config)
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

        save_file = "~/.hypergan/saves/"+args.config+".ckpt"
        samples_path = "~/.hypergan/samples/"+args.config+'/'
        self.create_path(save_file)
        self.create_path(samples_path)

        self.gan.load_or_initialize_graph(save_file)
        tf.train.start_queue_runners(sess=self.sess)

        self.output_graph_size()

        #TODO LOADING
        if args.method == 'train':
            self.train(args)
        elif args.method == 'serve':
            self.serve(args)
        elif args.method == 'build':
            self.build(args)


        tf.reset_default_graph()
        self.sess.close()

