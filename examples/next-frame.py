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

    def inputs(self):
        return self.frames

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


    def create(self):
        config = self.config
        ops = self.ops
        self.g_vars = []
        d_vars = []

        with tf.device(self.device):
            def random_t(shape):
                shape[-1] //= len(config.z_distribution.projections)
                return UniformEncoder(self, config.z_distribution, output_shape=shape).sample
            def random_like(x):
                shape = self.ops.shape(x)
                return random_t(shape)

            self.frame_count = len(self.inputs.frames)
            self.frames = self.inputs.frames

            dist = UniformEncoder(self, config.z_distribution)
            dist2 = UniformEncoder(self, config.z_distribution)
            dist3 = UniformEncoder(self, config.z_distribution)
            dist4 = UniformEncoder(self, config.z_distribution)
            uz = self.create_component(config.uz, name='u_to_z', input=dist.sample)
            uc = self.create_component(config.uc, name='u_to_c', input=dist2.sample)
            uz2 = self.create_component(config.uz, name='u_to_z', input=dist3.sample, reuse=True)
            uc2 = self.create_component(config.uc, name='u_to_c', input=dist4.sample, reuse=True)

            self.g_vars += uz.variables()
            self.g_vars += uc.variables()

            def ec(zt, cp,reuse=True):
                if config.noise:
                    randt = random_like(cp)
                    if config.proxy:
                        dist3 = UniformEncoder(self, config.z_distribution)
                        proxy_c = self.create_component(config.proxy_c, name='rand_ct', input=dist3.sample, reuse=reuse)
                        randt = proxy_c.sample
                    cp = tf.concat([cp, randt], axis=3)

                #cp = tf.concat([cp, random_like(cp)], axis=3)
                c = self.create_component(config.ec, name='ec', input=zt, features=[cp], reuse=reuse)
                if not reuse:
                    if config.proxy:
                        self.g_vars += proxy_c.variables()
                    self.g_vars += c.variables()
                return c.sample
            def ez(ft, zp,reuse=True):
                z = self.create_component(config.ez, name='ez', input=ft, features=[zp], reuse=reuse)
                if not reuse:
                    self.g_vars += z.variables()
                return z.sample

            def build_g(zt, ct, reuse=True):
                print("Gb", reuse)
                g = self.create_component(config.generator, name='generator', input=ct, features=[zt], reuse=reuse)
                if not reuse:
                    self.g_vars += g.variables()
                return g.sample

            def encode_frames(fs, c0, z0, reuse=True):
                cs = [c0]
                zs = [z0]
                x_hats = [build_g(zs[-1], cs[-1], reuse=reuse)]
                for i in range(len(fs)):
                    print("encode frames", i)
                    _reuse = reuse or (i!=0)
                    z = ez(fs[i], zs[-1], reuse=_reuse)
                    c = ec(z, cs[-1], reuse=_reuse)
                    x_hat = build_g(z, c, reuse=True)
                    zs.append(z)
                    cs.append(c)
                    x_hats.append(x_hat)
                return cs, zs, x_hats

            def build_sim(z0, c0, steps, reuse=True):
                zs = [z0]
                cs = [c0]
                gs = [build_g(zs[-1], cs[-1], reuse=reuse)]
                for i in range(steps):
                    _reuse = reuse or (i!=0)
                    z = ez(gs[-1], zs[-1], reuse=_reuse)
                    c = ec(z, cs[-1], reuse=_reuse)
                    g = build_g(z, c, reuse=True)
                    zs.append(z)
                    cs.append(c)
                    gs.append(g)

                return gs, cs, zs

            cs, zs, x_hats = encode_frames(self.frames, uc2.sample, uz2.sample, reuse=False)
            self.zs = zs
            self.cs = cs
            ugs, ucs, uzs = build_sim(uz.sample, uc.sample, len(self.frames)-1)
            ugs_next, ucs_next, uzs_next = build_sim(uzs[-1], ucs[-1], len(self.frames)-1)
            re_ucs_next, re_uzs_next, re_ugs_next = encode_frames(ugs_next[1:], ucs_next[0], uzs_next[0])
            gs_next, cs_next, zs_next = build_sim(zs[-1], cs[-1], len(self.frames)-1)
            re_ucs, re_uzs, ugs_hat = encode_frames(ugs[1:], ucs[0], uzs[0])
            re_cs_next, re_zs_next, re_gs_next = encode_frames(gs_next[1:], cs_next[0], zs_next[0])

            t0 = tf.concat(zs[2:], axis=3)
            t1 = tf.concat(re_uzs[1:], axis=3)
            t2 = tf.concat(re_zs_next[1:], axis=3)
            t3 = tf.concat(re_uzs_next[1:], axis=3)
            f0 = tf.concat(cs[1:-1], axis=3)
            f1 = tf.concat(re_ucs[:-1], axis=3)
            f2 = tf.concat(re_cs_next[:-1], axis=3)
            f3 = tf.concat(re_ucs_next[1:-1], axis=3)

            stack = [t0,t1, t2]#, t4, t5]
            stacked = ops.concat(stack, axis=0)
            features =ops.concat([f0,f1,f2], axis=0)
            d = self.create_component(config.z_discriminator, name='d_img', input=stacked, features=[features])
            d_vars += d.variables()
            l = self.create_loss(config.loss, d, None, None, len(stack))
            d_loss = l.d_loss
            g_loss = l.g_loss

            self.video_generator_last_z = uzs[0]
            self.video_generator_last_c = ucs[0]
            self.gs_next = gs_next
            ztn = uzs[1]
            ctn = ucs[1]
            self.video_generator_last_zn = ztn
            self.video_generator_last_cn = ctn
            gen = hc.Config({"sample":ugs[0]})
            if config.use_x:
                t0 = tf.concat(self.frames[1:], axis=3)
                t2 = tf.concat(ugs[:-1], axis=3)
                t7 = tf.concat(ugs_next, axis=3)
                t3 = tf.concat(gs_next[:-1], axis=3)
                t4 = tf.concat(self.frames[3:]+gs_next[:2], axis=3)
                t5 = tf.concat(self.frames[4:]+gs_next[:3], axis=3)
                t8 = tf.concat(self.frames[5:]+gs_next[:4], axis=3)
                t6 = tf.concat(self.frames[2:]+[gs_next[0]], axis=3)

                f0 = tf.concat(cs[1:-1], axis=3)
                f2 = tf.concat(ucs[:-1], axis=3)
                f7 = tf.concat(ucs_next, axis=3)
                f3 = tf.concat(cs_next[:-1], axis=3)
                f4 = tf.concat(cs[3:]+[cs_next[0]], axis=3)
                f5 = tf.concat(cs[4:]+cs_next[0:2], axis=3)
                f8 = tf.concat(cs[5:]+cs_next[0:3], axis=3)
                f6 = tf.concat(cs[2:], axis=3)
 
                stack = [t0]
                features = [f0]
                if config.encode_forward:
                    stack += [t3,t4,t5,t6]
                    features += [f3,f4,f5,f6]
                if config.encode_ug:
                    stack += [t2]
                    features += [f2]

                stacked = ops.concat(stack, axis=0)
                features = tf.concat(features, axis=0)
                d = self.create_component(config.discriminator, name='d_manifold', input=stacked, features=[features])
                d_vars += d.variables()
                l = self.create_loss(config.loss, d, None, None, len(stack))
                d_loss += l.d_loss
                g_loss += l.g_loss

    
            gx_sample = gen.sample
            gy_sample = gen.sample
            gx = hc.Config({"sample":gx_sample})
            gy = hc.Config({"sample":gy_sample})

            last_frame = tf.slice(gy_sample, [0,0,0,0], [-1, -1, -1, 3])
            self.y = hc.Config({"sample":last_frame})
            self.gy = self.y
            self.gx = self.y
            self.uniform_sample = gen.sample

            self.preview = tf.concat(self.inputs.frames[:-1] + [gen.sample], axis=1)#tf.concat(tf.split(gen.sample, (self.ops.shape(gen.sample)[3]//3), 3), axis=1)


            metrics = {
                    'g_loss': g_loss,
                    'd_loss': d_loss
                }

 
            trainers = []

            lossa = hc.Config({'sample': [d_loss, g_loss], 'metrics': metrics, 'd_fake': l.d_fake, 'd_real': l.d_real, 'config': l.config})
            self.loss = lossa
            self._g_vars = self.g_vars
            self._d_vars = d_vars
            trainer = self.create_component(config.trainer, loss = lossa, g_vars = self.g_vars, d_vars = d_vars)
            self.session.run(tf.global_variables_initializer())

        self.trainer = trainer
        self.generator = gx
        self.z_hat = gy.sample
        self.x_input = self.inputs.frames[0]

        self.uga = self.y.sample
        self.uniform_encoder = dist

    def g_vars(self):
        return self._g_vars
    def d_vars(self):
        return self._d_vars

    def fitness_inputs(self):
        return self.inputs.frames

    def create_loss(self, loss_config, discriminator, x, generator, split):
        loss = self.create_component(loss_config, discriminator = discriminator, x=x, generator=generator, split=split)
        return loss

    def create_encoder(self, x_input, name='input_encoder', reuse=False):
        config = self.config
        input_encoder = dict(config.input_encoder or config.g_encoder or config.generator)
        encoder = self.create_component(input_encoder, name=name, input=x_input, reuse=reuse)
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
        self.x = gan.session.run(gan.preview)
        print("__________", np.shape(self.x),'---oo')
        frames = np.shape(self.x)[1]//height
        self.frames=frames
        self.x = np.split(self.x, frames, axis=1)
        self.i = 0
        BaseSampler.__init__(self, gan, samples_per_row)

    def _sample(self):
        gan = self.gan
        z_t = gan.uniform_encoder.sample
        sess = gan.session

        feed_dict = {}
        for i,f in enumerate(gan.inputs.frames):
            if len(self.x) > i+1:
                feed_dict[f]=self.x[i+1]
            #if(1 + self.frames < len(self.x)):
            #    feed_dict[f] = self.x[1+self.frames]
        self.x = sess.run(gan.preview, feed_dict)
        frames = np.shape(self.x)[1]//height
        self.x = np.split(self.x, frames, axis=1)
        x_ = self.x[-1]

        time.sleep(0.15)
        return {
            'generator': x_
        }


class TrainingVideoFrameSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        self.z = None

        self.i = 0
        BaseSampler.__init__(self, gan, samples_per_row)

    def _sample(self):
        gan = self.gan
        z_t = gan.uniform_encoder.sample
        sess = gan.session
        
 
        return {
            'generator': gan.session.run(gan.preview)
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
