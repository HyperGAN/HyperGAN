import os
import uuid
import random
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import numpy as np
import glob
from hypergan.viewer import GlobalViewer
from hypergan.samplers.base_sampler import BaseSampler
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.samplers.random_walk_sampler import RandomWalkSampler
from hypergan.samplers.debug_sampler import DebugSampler
from hypergan.search.alphagan_random_search import AlphaGANRandomSearch
from hypergan.gans.base_gan import BaseGAN
from common import *

from hypergan.gans.alpha_gan import AlphaGAN

arg_parser = ArgumentParser("render next frame")
arg_parser.add_image_arguments()
args = arg_parser.parse_args()

width, height, channels = parse_size(args.size)

config = lookup_config(args)
if args.action == 'search':
    random_config = AlphaGANRandomSearch({}).random_config()
    if args.config_list is not None:
        config = random_config_from_list(args.config_list)

        config["generator"]=random_config["generator"]
        config["g_encoder"]=random_config["g_encoder"]
        config["discriminator"]=random_config["discriminator"]
        config["z_discriminator"]=random_config["z_discriminator"]

        # TODO Other search terms?
    else:
        config = random_config

def add_prev_frames_g(gan, config, net):
    x1 = gan.layer_1
    x2 = gan.layer_2
    print("!!layer", x1, x2)
    s = [int(x) for x in net.get_shape()]
    s[3]*=2
    print("S IS ", s)
    shape = [s[1], s[2]]
    x = tf.concat([x1,x2], axis=3)
    x = tf.image.resize_images(x, shape, 1)

    return x

config['generator']['layer_filter']=add_prev_frames_g

class VideoFrameLoader:
    """
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def create(self, directory, channels=3, format='jpg', width=64, height=64, crop=False, resize=False):
        directories = glob.glob(directory+"/*")
        directories = [d for d in directories if os.path.isdir(d)]

        if(len(directories) == 0):
            directories = [directory] 

        # Create a queue that produces the filenames to read.
        if(len(directories) == 1):
            # No subdirectories, use all the images in the passed in path
            filenames = glob.glob(directory+"/*."+format)
        else:
            filenames = glob.glob(directory+"/**/*."+format)

        self.file_count = len(filenames)
        filenames = sorted(filenames)
        print("FILENAMES", filenames[:-2][0], filenames[1:-1][0], filenames[2:-2][0])
        if self.file_count == 0:
            raise ValidationException("No images found in '" + directory + "'")


        input_t = [filenames[:-2], filenames[1:-1], filenames[2:]]
        #input_t = [f1 + ',' + f2 + ',' + f3 for f1,f2,f3 in zip(*input_t)]
        #input_queue = tf.train.string_input_producer(input_t, shuffle=True)
        #x1,x2,x3 = tf.decode_csv(input_queue.dequeue(), [[""], [""], [""]], ",")
        input_queue = tf.train.slice_input_producer(input_t, shuffle=True)
        x1,x2,x3 = input_queue
        print('---',x1)

        # Read examples from files in the filename queue.
        x1 = self.read_frame(x1, format, crop, resize)
        x2 = self.read_frame(x2, format, crop, resize)
        x3 = self.read_frame(x3, format, crop, resize)
        x1,x2,x3 = self._get_data(x1,x2,x3)
        self.x1 = self.x = x1
        self.x2 = x2
        self.x3 = x3
        return [x1, x2, x3], None


    def read_frame(self, t, format, crop, resize):
        value = tf.read_file(t)

        if format == 'jpg':
            img = tf.image.decode_jpeg(value, channels=channels)
        elif format == 'png':
            img = tf.image.decode_png(value, channels=channels)
        else:
            print("[loader] Failed to load format", format)
        img = tf.cast(img, tf.float32)


      # Image processing for evaluation.
      # Crop the central [height, width] of the image.
        if crop:
            resized_image = hypergan.inputs.resize_image_patch.resize_image_with_crop_or_pad(img, height, width, dynamic_shape=True)
        elif resize:
            resized_image = tf.image.resize_images(img, [height, width], 1)
        else: 
            resized_image = img

        tf.Tensor.set_shape(resized_image, [height,width,channels])

        # This moves the image to a range of -1 to 1.
        float_image = resized_image / 127.5 - 1.

        return float_image

    def _get_data(self, x1,x2,x3):
        batch_size = self.batch_size
        num_preprocess_threads = 24
        x1,x2,x3 = tf.train.shuffle_batch(
            [x1,x2,x3],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity= batch_size*2, min_after_dequeue=batch_size)
        return x1,x2,x3
inputs = VideoFrameLoader(args.batch_size)
inputs.create(args.directory,
        channels=channels, 
        format=args.format,
        crop=args.crop,
        width=width,
        height=height,
        resize=True)

save_file = "save/model.ckpt"

class NextFrameGAN(BaseGAN):
    """ 
    """
    def __init__(self, *args, **kwargs):
        self.discriminator = None
        self.encoder = None
        self.generator = None
        self.loss = None
        self.trainer = None
        self.session = None
        self.layer_1 = kwargs['inputs'].x1
        self.layer_2 = kwargs['inputs'].x2
        BaseGAN.__init__(self, *args, **kwargs)

    def required(self):
        return "generator".split()

    def create(self):
        config = self.config

        with tf.device(self.device):
            if self.session is None: 
                self.session = self.ops.new_session(self.ops_config)

            #this is in a specific order
            if self.encoder is None and config.encoder:
                self.encoder = self.create_component(config.encoder)
                self.uniform_encoder = self.encoder
            if self.generator is None and config.generator:
                self.generator = self.create_component(config.generator)
                self.autoencoded_x = self.generator.sample
                self.uniform_sample = self.generator.sample

            if self.discriminator is None and config.discriminator:
                x1 = self.inputs.x1
                x2 = self.inputs.x2
                x3 = self.inputs.x3
                g = self.generator.sample
                real_frames = [x1,x2,x3]
                gen_frames = [x1,x2,g]
                real_frames = tf.concat(real_frames, axis=3)
                gen_frames = tf.concat(gen_frames, axis=3)
                stacked = [real_frames, gen_frames]
                stacked_xg = tf.concat(stacked, axis=0)
                self.discriminator = self.create_component(config.discriminator, name="discriminator", input=stacked_xg)
            if self.loss is None and config.loss:
                self.loss = self.create_component(config.loss)
                self.metrics = self.loss.metrics
            if self.trainer is None and config.trainer:
                self.trainer = self.create_component(config.trainer)

            self.random_z = tf.random_uniform(self.ops.shape(self.encoder.sample), -1, 1, name='random_z')

            self.session.run(tf.global_variables_initializer())

    def input_nodes(self):
        "used in hypergan build"
        return [
                self.uniform_encoder.sample
        ]

    def output_nodes(self):
        "used in hypergan build"
        return [
                self.uniform_sample,
                self.random_z
        ]

class VideoFrameSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        self.z = None

        #self.last_frame_1 = gan.session.run(tf.ones_like(gan.inputs.x)*-1)
        #self.last_frame_2 = gan.session.run(tf.ones_like(gan.inputs.x)*-1)
        self.last_frame_1 = gan.session.run(gan.inputs.x1)
        self.last_frame_2 = gan.session.run(gan.inputs.x2)
        #self.last_frame_1 = gan.session.run(tf.zeros_like(gan.inputs.x))
        #self.last_frame_2 = gan.session.run(tf.zeros_like(gan.inputs.x))

        self.i = 0
        BaseSampler.__init__(self, gan, samples_per_row)

    def _sample(self):
        gan = self.gan
        z_t = gan.uniform_encoder.sample
        next_frame = gan.session.run(gan.g_blank3, {gan.g_blank2: self.last_frame_2, gan.g_blank: self.last_frame_1})
        self.last_frame_1 = self.last_frame_2
        self.last_frame_2 = next_frame
        self.i += 1
        if self.i > 600:
            self.last_frame_1 = np.zeros_like(self.last_frame_1)
            self.last_frame_2 = np.zeros_like(self.last_frame_1)
            self.last_frame_1 = gan.session.run(gan.inputs.x1)
            self.last_frame_2 = gan.session.run(gan.inputs.x2)
            self.i = 0

        return {
            'generator': next_frame
        }



class ProgressiveNextFrameGAN(BaseGAN):
    """ 
    """
    def __init__(self, *args, **kwargs):
        self.discriminator = None
        self.encoder = None
        self.generator = None
        self.loss = None
        self.trainer = None
        self.session = None
        self.layer_1 = kwargs['inputs'].x1
        self.layer_2 = kwargs['inputs'].x2

        BaseGAN.__init__(self, *args, **kwargs)

    def required(self):
        return "generator".split()

    def create(self):
        config = self.config

        with tf.device(self.device):
            if self.session is None: 
                self.session = self.ops.new_session(self.ops_config)

            #this is in a specific order
            if self.encoder is None and config.encoder:
                self.encoder = self.create_component(config.encoder)
                self.uniform_encoder = self.encoder

            l1 = self.inputs.x1
            l2 = self.inputs.x2
            shape = [int(x) for x in self.inputs.x2.get_shape()]
            if self.generator is None and config.generator:
                self.generator = self.create_component(config.generator)
                self.autoencoded_x = self.generator.sample

            if self.discriminator is None and config.discriminator:
                x1 = self.inputs.x1
                x2 = self.inputs.x2
                x3 = self.inputs.x3

                self.layer_1 = tf.zeros_like(x1)
                self.layer_2 = tf.zeros_like(x2)
                g_blank = self.generator.reuse(net=self.uniform_encoder.create())
                self.layer_1 = tf.zeros_like(x1)
                self.layer_2 = g_blank
                g_blank2 = self.generator.reuse(net=self.uniform_encoder.create())
                self.layer_1 = g_blank
                self.layer_2 = g_blank2
                g_blank3 = self.generator.reuse(net=self.uniform_encoder.create())
                self.layer_1 = g_blank2
                self.layer_2 = g_blank3
                g_blank4 = self.generator.reuse(net=self.uniform_encoder.create())
                self.layer_1 = g_blank3
                self.layer_2 = g_blank4
                g_blank5 = self.generator.reuse(net=self.uniform_encoder.create())
                self.layer_1 = g_blank4
                self.layer_2 = g_blank5
                g_blank6 = self.generator.reuse(net=self.uniform_encoder.create())
                gen_blank = [g_blank, g_blank2, g_blank3]
                gen_blank2 = [g_blank4, g_blank5, g_blank6]
                gen_blank_frames = tf.concat(gen_blank, axis=3)
                gen_blank2_frames = tf.concat(gen_blank2, axis=3)
                #self.layer_1 = tf.assign(self.layer_1, x1)
                #self.layer_2 = tf.assign(self.layer_2, x2)
                self.layer_1 = x1
                self.layer_2 = x2
                g = self.generator.reuse(net=self.uniform_encoder.create())
                real_frames = [x1,x2,x3]
                gen1_frames = [x1,x2,g]
                real_frames = tf.concat(real_frames, axis=3)
                gen1_frames = tf.concat(gen1_frames, axis=3)
                #self.layer_1 = tf.assign(self.layer_1, x2)
                #self.layer_2 = tf.assign(self.layer_2, g)
                self.layer_1 = x2
                self.layer_2 = g
                g2 = self.generator.reuse(net=self.uniform_encoder.create())
                gen2_frames = tf.concat([x2,g,g2], axis=3)
                #self.layer_1 = tf.assign(self.layer_1, g)
                #self.layer_2 = tf.assign(self.layer_2, g2)
                self.layer_1 = g
                self.layer_2 = g2
                g3 = self.generator.reuse(net=self.uniform_encoder.create())
                gen3_frames = tf.concat([g,g2,g3], axis=3)
                stacked = [real_frames, gen1_frames, gen2_frames, gen3_frames, gen_blank_frames, gen_blank2_frames]
                self.g_blank3 = g_blank3
                self.g_blank2 = g_blank2
                self.g_blank = g_blank
                self.seq = [x1,x2, x3,g,g2,g3, g_blank, g_blank2, g_blank3, g_blank4, g_blank5, g_blank6]
                stacked_xg = tf.concat(stacked, axis=0)
                self.next_frame=g3
                self.discriminator = self.create_component(config.discriminator, name="discriminator", input=stacked_xg)
            if self.loss is None and config.loss:
                self.uniform_sample = g3 #HACK for rothk
                self.loss = self.create_component(config.loss, split=len(stacked))
                #self.uniform_sample = g_random
                #xg_stack = tf.concat([x1,g_random], axis=0)
                #d2 = self.create_component(config.discriminator, name='d2', input=xg_stack)
                #loss2 = self.create_component(config.loss, discriminator=d2)
                #print("D_LOSS", loss2, loss2.d_loss, loss2.g_loss)
                #self.discriminator = d2
                #self.loss.sample[0] += loss2.d_loss
                #self.loss.sample[1] += loss2.g_loss
                #print("ADDING")
                self.metrics = self.loss.metrics
            if self.trainer is None and config.trainer:
                self.trainer = self.create_component(config.trainer, d_vars = self.discriminator.variables())
                #self.trainer = self.create_component(config.trainer, d_vars = self.discriminator.variables())# + d2.variables())

            self.random_z = tf.random_uniform(self.ops.shape(self.encoder.sample), -1, 1, name='random_z')

            self.session.run(tf.global_variables_initializer())

    def input_nodes(self):
        "used in hypergan build"
        return [
                self.uniform_encoder.sample
        ]

    def output_nodes(self):
        "used in hypergan build"
        return [
                self.random_z
        ]


def setup_gan(config, inputs, args):
    gan = ProgressiveNextFrameGAN(config, inputs=inputs)

    if(args.action != 'search' and os.path.isfile(save_file+".meta")):
        gan.load(save_file)

    tf.train.start_queue_runners(sess=gan.session)

    config_name = args.config
    GlobalViewer.title = "[hypergan] next-frame " + config_name
    GlobalViewer.enabled = args.viewer
    GlobalViewer.zoom = 1

    return gan

def train(config, inputs, args):
    gan = setup_gan(config, inputs, args)
    sampler = lookup_sampler(args.sampler or DebugSampler)(gan)
    samples = 0

    metrics = [batch_accuracy(gan.inputs.x, gan.uniform_sample), batch_diversity(gan.uniform_sample)]
    sum_metrics = [0 for metric in metrics]
    for i in range(args.steps):
        gan.step()

        if args.action == 'train' and i % args.save_every == 0 and i > 0:
            print("saving " + save_file)
            gan.save(save_file)

        if i % args.sample_every == 0:
            sample_file="samples/%06d.png" % (samples)
            samples += 1
            sampler.sample(sample_file, args.save_samples)

        if i > args.steps * 9.0/10:
            for k, metric in enumerate(gan.session.run(metrics)):
                print("Metric "+str(k)+" "+str(metric))
                sum_metrics[k] += metric 

    tf.reset_default_graph()
    return sum_metrics

def sample(config, inputs, args):
    gan = setup_gan(config, inputs, args)
    sampler = lookup_sampler(args.sampler or VideoFrameSampler)(gan)
    samples = 0
    for i in range(args.steps):
        sample_file="samples/%06d.png" % (samples)
        samples += 1
        sampler.sample(sample_file, args.save_samples)

def search(config, inputs, args):
    metrics = train(config, inputs, args)

    config_filename = "colorizer-"+str(uuid.uuid4())+'.json'
    hc.Selector().save(config_filename, config)
    with open(args.search_output, "a") as myfile:
        myfile.write(config_filename+","+",".join([str(x) for x in metrics])+"\n")

if args.action == 'train':
    metrics = train(config, inputs, args)
    print("Resulting metrics:", metrics)
elif args.action == 'sample':
    sample(config, inputs, args)
elif args.action == 'search':
    search(config, inputs, args)
else:
    print("Unknown action: "+args.action)
