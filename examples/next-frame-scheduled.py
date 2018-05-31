import os
import uuid
import random
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import numpy as np
import glob
import time
import re
from hypergan.viewer import GlobalViewer
from hypergan.samplers.base_sampler import BaseSampler
from hypergan.gan_component import ValidationException, GANComponent
from hypergan.samplers.random_walk_sampler import RandomWalkSampler
from hypergan.samplers.debug_sampler import DebugSampler
from hypergan.search.alphagan_random_search import AlphaGANRandomSearch
from hypergan.gans.base_gan import BaseGAN
from common import *

import copy

from hypergan.gans.alpha_gan import AlphaGAN

from hypergan.gan_component import ValidationException, GANComponent
from hypergan.gans.base_gan import BaseGAN

from hypergan.discriminators.fully_connected_discriminator import FullyConnectedDiscriminator
from hypergan.encoders.uniform_encoder import UniformEncoder
from hypergan.trainers.multi_step_trainer import MultiStepTrainer
from hypergan.trainers.multi_trainer_trainer import MultiTrainerTrainer
from hypergan.trainers.consensus_trainer import ConsensusTrainer
from hypergan.trainers.incremental_trainer import IncrementalTrainer


arg_parser = ArgumentParser("render next frame")
parser = arg_parser.add_image_arguments()
parser.add_argument('--frames', type=int, default=4, help='Number of frames to embed.')
parser.add_argument('--shuffle', type=bool, default=False, help='Randomize inputs.')
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


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

class VideoFrameLoader:
    """
    """

    def __init__(self, batch_size, frame_count, shuffle):
        self.batch_size = batch_size
        self.frame_count = frame_count
        self.shuffle = shuffle

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

        if(len(filenames) < self.frame_count):
            print("Error: Not enough frames in data folder ", directory)

        self.file_count = len(filenames)
        filenames = sorted(filenames, key=alphanum_key)
        if self.file_count == 0:
            raise ValidationException("No images found in '" + directory + "'")


        # creates arrays of filenames[:end], filenames[1:end-1], etc for serialized random batching
        if self.shuffle:
            frames  = [tf.train.slice_input_producer([filenames], shuffle=True)[0] for i in range(self.frame_count)]
        else:
            input_t = [filenames[i:i-self.frame_count] for i in range(self.frame_count)]
            input_queue = tf.train.slice_input_producer(input_t, shuffle=True)
            frames = input_queue

        # Read examples from files in the filename queue.
        frames = [self.read_frame(frame, format, crop, resize) for frame in frames]
        frames = self._get_data(frames)
        self.frames = frames

        x  = tf.train.slice_input_producer([filenames], shuffle=True)[0]
        y  = tf.train.slice_input_producer([filenames], shuffle=True)[0]
        self.x = self.read_frame(x, format, crop, resize)
        self.y = self.read_frame(y, format, crop, resize)
        self.x = self._get_data([self.x])
        self.y = self._get_data([self.y])


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

    def _get_data(self, imgs):
        batch_size = self.batch_size
        num_preprocess_threads = 24
        return tf.train.shuffle_batch(
                imgs,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity= batch_size*2, min_after_dequeue=batch_size)
inputs = VideoFrameLoader(args.batch_size, args.frames, args.shuffle)
inputs.create(args.directory,
        channels=channels, 
        format=args.format,
        crop=args.crop,
        width=width,
        height=height,
        resize=True)

save_file = "save/model.ckpt"

class AliNextFrameGAN(BaseGAN):
    """ 
    """
    def __init__(self, *args, **kwargs):
        BaseGAN.__init__(self, *args, **kwargs)

    def required(self):
        """
        `input_encoder` is a discriminator.  It encodes X into Z
        `discriminator` is a standard discriminator.  It measures X, reconstruction of X, and G.
        `generator` produces two samples, input_encoder output and a known random distribution.
        """
        return "generator discriminator ".split()

    def create(self):
        config = self.config
        ops = self.ops

        with tf.device(self.device):
            def random_like(x):
                shape = self.ops.shape(x)
                shape[-1] //= len(config.z_distribution.projections)
                return UniformEncoder(self, config.z_distribution, output_shape=shape).sample
            self.frame_count = len(self.inputs.frames)
            self.frames = self.inputs.frames

            self.y = hc.Config({"sample": self.frames[0]})

            style = self.create_component(config.style_encoder, input=self.inputs.y, name='xb_style')
            self.style = style

            d = self.create_component(config.discriminator, name='d_ab3', input=self.frames[0], features=[None])
            ue = random_like(d.sample)
            print("UE", ue)
            def generator(c, zs):
                def g(z, reuse=False):
                    return self.create_component(c, features=[], input=z, name='g', reuse=reuse)

                return [g(zs[0])]+[g(z, reuse=True) for z in zs[1:]]

            def next_z(c, zs):
                z = tf.concat(zs, len(self.ops.shape(zs[0]))-1)
                return self.create_component(c, features=[], input=z, name='z')

            z_frames = [random_like(ue) for frame in self.frames]
            print("z_frames", z_frames)
            g_frames = generator(config.generator, z_frames)
            self.z_frames = z_frames
            self.g_frames = g_frames

            self.uniform_encoder = ue
            self.uniform_sample = g_frames[0].sample

            t0 = self.frames[0]
            t1 = g_frames[0].sample
            stack = [t0, t1]
            stacked = ops.concat(stack, axis=0)
            features = None#ops.concat([f0, f1, f2], axis=0)

            d = self.create_component(config.discriminator, name='d_ab', input=stacked, features=[features])
            l = self.create_loss(config.loss, d, None, None, len(stack))
            loss1 = l
            d_loss1 = l.d_loss
            g_loss1 = l.g_loss

            def reuse_d(xs):
                xs_concat = tf.concat(xs, axis=0)
                d = self.create_component(config.discriminator, name='d_ab', input=xs_concat, features=[None], reuse=True)
                if 'fc7' in d.controls:
                    ds = self.split_batch(d.controls['fc7'], len(xs))
                else:
                    ds = self.split_batch(d.sample, len(xs))
                return ds
            ez_frames = reuse_d(self.frames)
            next_z = next_z(config.next_z, ez_frames[:-1]+[random_like(z_frames[0])])
            self.next_z = next_z
            self.next_g = self.create_component(config.generator, features=[], input=next_z.sample, name='g', reuse=True)

            d_vars1 = d.variables()
            g_vars1 = g_frames[0].variables()
            if config.style:
                g_vars1 += style.variables()

            metrics = {
                    'g_loss': l.g_loss,
                    'd_loss': l.d_loss
                    }


            axis = len(ops.shape(z_frames[0]))-1
            t0 = tf.concat(ez_frames, axis=axis)
            t1 = tf.concat(ez_frames[:-1] + [next_z.sample], axis=axis)
            stack = [t0, t1]
            stacked = ops.concat(stack, axis=0)
            features = None#ops.concat([f0, f1, f2], axis=0)

            d2 = self.create_component(config.z_discriminator, name='d_ab2', input=stacked, features=[features])
            l2 = self.create_loss(config.loss, d2, None, None, len(stack))
            loss2 = l2
            d_loss2 = l2.d_loss
            g_loss2 = l2.g_loss

            d_vars2 = d2.variables()
            g_vars2 = next_z.variables()

            metrics2 = {
                    'next_g': l2.g_loss,
                    'next_d': l2.d_loss
                }


            lossa = hc.Config({'sample': [d_loss1, g_loss1], 'metrics': metrics})
            lossb = hc.Config({'sample': [d_loss2, g_loss2], 'metrics': metrics2})
            trainer1 = ConsensusTrainer(self, config.trainer, loss = lossa, g_vars = g_vars1, d_vars = d_vars1)
            trainer2 = ConsensusTrainer(self, config.trainer, loss = lossb, g_vars = g_vars2, d_vars = d_vars2)
            trainer = IncrementalTrainer([trainer1, trainer2], config.training_schedule or [10000, 10000])
            self.session.run(tf.global_variables_initializer())

        self.trainer = trainer
        self.generator = self.g_frames[0]
        self.x_input = self.inputs.frames[0]

        self.uga = self.y.sample



    def create_loss(self, loss_config, discriminator, x, generator, split):
        loss = self.create_component(loss_config, discriminator = discriminator, x=x, generator=generator, split=split)
        return loss

    def create_encoder(self, x_input, name='input_encoder'):
        config = self.config
        input_encoder = dict(config.input_encoder or config.g_encoder or config.generator)
        encoder = self.create_component(input_encoder, name=name, input=x_input)
        return encoder

    def create_z_discriminator(self, z, z_hat):
        config = self.config
        z_discriminator = dict(config.z_discriminator or config.discriminator)
        z_discriminator['layer_filter']=None
        net = tf.concat(axis=0, values=[z, z_hat])
        encoder_discriminator = self.create_component(z_discriminator, name='z_discriminator', input=net)
        return encoder_discriminator

    def create_cycloss(self, x_input, x_hat):
        config = self.config
        ops = self.ops
        distance = config.distance or ops.lookup('l1_distance')
        pe_layers = self.gan.skip_connections.get_array("progressive_enhancement")
        cycloss_lambda = config.cycloss_lambda
        if cycloss_lambda is None:
            cycloss_lambda = 10
        
        if(len(pe_layers) > 0):
            mask = self.progressive_growing_mask(len(pe_layers)//2+1)
            cycloss = tf.reduce_mean(distance(mask*x_input,mask*x_hat))

            cycloss *= mask
        else:
            cycloss = tf.reduce_mean(distance(x_input, x_hat))

        cycloss *= cycloss_lambda
        return cycloss


    def create_z_cycloss(self, z, x_hat, encoder, generator):
        config = self.config
        ops = self.ops
        total = None
        distance = config.distance or ops.lookup('l1_distance')
        if config.z_hat_lambda:
            z_hat_cycloss_lambda = config.z_hat_cycloss_lambda
            recode_z_hat = encoder.reuse(x_hat)
            z_hat_cycloss = tf.reduce_mean(distance(z_hat,recode_z_hat))
            z_hat_cycloss *= z_hat_cycloss_lambda
        if config.z_cycloss_lambda:
            recode_z = encoder.reuse(generator.reuse(z))
            z_cycloss = tf.reduce_mean(distance(z,recode_z))
            z_cycloss_lambda = config.z_cycloss_lambda
            if z_cycloss_lambda is None:
                z_cycloss_lambda = 0
            z_cycloss *= z_cycloss_lambda

        if config.z_hat_lambda and config.z_cycloss_lambda:
            total = z_cycloss + z_hat_cycloss
        elif config.z_cycloss_lambda:
            total = z_cycloss
        elif config.z_hat_lambda:
            total = z_hat_cycloss
        return total



    def input_nodes(self):
        "used in hypergan build"
        if hasattr(self.generator, 'mask_generator'):
            extras = [self.mask_generator.sample]
        else:
            extras = []
        return extras + [
                self.x_input
        ]


    def output_nodes(self):
        "used in hypergan build"

    
        if hasattr(self.generator, 'mask_generator'):
            extras = [
                self.mask_generator.sample, 
                self.generator.g1x,
                self.generator.g2x
            ]
        else:
            extras = []
        return extras + [
                self.encoder.sample,
                self.generator.sample, 
                self.uniform_sample,
                self.generator_int
        ]
class VideoFrameSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        sess = gan.session

        self.z_frames = gan.session.run(gan.z_frames[:-1])
        self.next_z = gan.session.run(gan.next_z.sample)
        self.i = 0
        BaseSampler.__init__(self, gan, samples_per_row)

    def _sample(self):
        gan = self.gan
        sess = gan.session

        feed_dict = dict(zip(gan.z_frames, self.z_frames))
        sample, next_z = sess.run([gan.next_g.sample, gan.next_z.sample], feed_dict=feed_dict)
        self.z_frames = self.z_frames[1:] + [next_z]

        time.sleep(0.05)
        return {


            'generator': sample
        }


class TrainingVideoFrameSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        self.z = None

        self.i = 0
        BaseSampler.__init__(self, gan, samples_per_row)

    def _sample(self):
        gan = self.gan
        sess = gan.session
 
        return {
            'generator': sess.run(gan.generator.sample)
        }




def setup_gan(config, inputs, args):
    gan = AliNextFrameGAN(config, inputs=inputs)

    if(args.action != 'search' and os.path.isfile(save_file+".meta")):
        gan.load(save_file)

    tf.train.start_queue_runners(sess=gan.session)

    config_name = args.config
    GlobalViewer.title = "[hypergan] next-frame " + config_name
    GlobalViewer.enabled = args.viewer
    GlobalViewer.zoom = args.zoom

    return gan

def train(config, inputs, args):
    gan = setup_gan(config, inputs, args)
    sampler = lookup_sampler(args.sampler or TrainingVideoFrameSampler)(gan)
    samples = 0

    #metrics = [batch_accuracy(gan.inputs.x, gan.uniform_sample), batch_diversity(gan.uniform_sample)]
    #sum_metrics = [0 for metric in metrics]
    for i in range(args.steps):
        gan.step()

        if args.action == 'train' and i % args.save_every == 0 and i > 0:
            print("saving " + save_file)
            gan.save(save_file)

        if i % args.sample_every == 0:
            sample_file="samples/%06d.png" % (samples)
            samples += 1
            sampler.sample(sample_file, args.save_samples)

        #if i > args.steps * 9.0/10:
        #    for k, metric in enumerate(gan.session.run(metrics)):
        #        print("Metric "+str(k)+" "+str(metric))
        #        sum_metrics[k] += metric 

    tf.reset_default_graph()
    return []#sum_metrics

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
