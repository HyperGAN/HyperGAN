import os
import uuid
import random
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import numpy as np
import glob
import time
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
            #x_input = tf.identity(self.inputs.x, name='input')
            x_input = tf.identity(self.inputs.x3, name='xb_i')
            last_frame_1 = tf.identity(self.inputs.x1, name='x1_t')
            last_frame_2 = tf.identity(self.inputs.x2, name='x2_t')
            self.last_frame_1 = last_frame_1
            self.last_frame_2 = last_frame_2

            hack_gen = config.generator
            generator = self.create_component(hack_gen, input=x_input, name='a_generator')

            z = generator.controls["z"]
            self.z = z
            generator.reuse(last_frame_1)
            z1 = generator.controls["z"]
            self.z1 = z1
            generator.reuse(last_frame_2)
            z2 = generator.controls["z"]
            self.z2 = z2


            trash = self.create_component(config.c, name='trash', input=z, features=[])

            def C(z, c, reuse=True, random=False):
                print("Z IS C", z, c, random)
                if(random):
                    z_shape = self.ops.shape(z)
                    uz_shape = z_shape
                    uz_shape[-1] = uz_shape[-1] // len(config.z_distribution.projections)
                    ue = UniformEncoder(self, config.z_distribution, output_shape=uz_shape)

                    noise = ue.sample
                    z = tf.concat(values=[noise, z], axis=3)
                print("ERUSE", reuse, z, c)
                c = self.create_component(config.c, name='c', input=z, features=[c], reuse=reuse)
                return c

            z_shape = self.ops.shape(trash.sample)
            uz_shape = z_shape

            c_t_rand2 = UniformEncoder(self, config.z_distribution, output_shape=uz_shape).sample
            if config.zeros:
                unrolled_c = C(z1, tf.zeros_like((c_t_rand2)), reuse=False, random=config.add_random)
            else:
                unrolled_c = C(z1, c_t_rand2, reuse=False, random=config.add_random)
            c_t_prev = C(z2, unrolled_c.sample, random=config.add_random, reuse=True).sample
            c2 = c_t_prev
            c_t_current = C(z, c_t_prev, random=config.add_random, reuse=True).sample

            uz_shape[-1] = uz_shape[-1] // len(config.z_distribution.projections)
            ue = UniformEncoder(self, config.z_distribution, output_shape=uz_shape)
            c_random = ue.sample

            def random_like(x):
                return UniformEncoder(self, config.z_distribution, output_shape=self.ops.shape(x)).sample

            self.c = c_t_current
            self.c2 = c_t_prev
            self.c1 = unrolled_c.sample
            self.c0 = c_t_rand2



            self.uniform_sample = generator.sample

            x_encoded = self.create_component(config.c, input=z2, features=[c2], name='x_encoder')
            uz_shape = z_shape
            uz_shape[-1] = uz_shape[-1] // len(config.z_distribution.projections)
            ue = UniformEncoder(self, config.z_distribution, output_shape=uz_shape)
            ue2 = UniformEncoder(self, config.z_distribution, output_shape=ops.shape(z))

            zua = ue.sample

            cz = self.create_component(config.c_to_z, name='c_to_z', input=ue.sample)
            ugb = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.sample})
            self.video_sample = ugb
            x_hat = generator.reuse(tf.zeros_like(x_input), replace_controls={"z": cz.reuse(c_t_prev)})
            self.x_next = generator.reuse(tf.zeros_like(x_input), replace_controls={"z": cz.reuse(c_t_current)})
            self.c_next = C(random_like(z), c_t_current, random=config.add_random, reuse=True).sample
            self.video_next =generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.reuse(self.c_next)})


            #f2 = cz.reuse(x_encoded.sample)
            #ga = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":z})
            #x_hat = generator.reuse(tf.zeros_like(x_input), replace_controls={"z": f2})
            if config.first_frame:
                t0 =x_input #self.last_frame_2
            else:
                t0 =self.last_frame_2
            t1 = ugb
            if config.first_frame:
                f0 = cz.reuse(c_t_prev)
            else:
                f0 = cz.reuse(c_t_current)
            f1 = cz.sample
            stack = [t0, t1]
            stacked = ops.concat(stack, axis=0)
            features = ops.concat([f0, f1], axis=0)
            if config.norandom:
                t0 = self.last_frame_2
                t1 = x_hat
                t2 = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":z2})
                f0 = cz.reuse(c_t_current)
                f1 = cz.reuse(c_t_prev)
                f2 = self.z2

                stack = [t0, t1, t2]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1, f2], axis=0)
            if config.current:
                x_next = generator.reuse(tf.zeros_like(x_input), replace_controls={"z": cz.reuse(c_t_current)})
                t0 = self.last_frame_1
                t1 = x_next
                f0 = cz.reuse(c_t_prev)
                f1 = cz.reuse(c_t_current)
                stack = [t0, t1]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1], axis=0)

            if config.dnobs:
                cz_prev = cz.reuse(c_t_prev)
                t0 = x_input #self.last_frame_2
                t1 = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz_prev})
                t2 = ugb
                f0 = cz_prev 
                f1 = cz.reuse(c_t_current)
                f2 = cz.sample
                stack = [t0, t1, t2]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1, f2], axis=0)

            noise = random_like(c_t_prev)
            cdist = self.create_component(config.u_to_c, name='cdist', input=noise)

            if config.dnobs2:
                cz_prev = cz.reuse(c_t_prev)
                t0 = x_input #self.last_frame_2
                t1 = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz_prev})
                t2 = ugb
                f0 = c_t_prev
                f1 = c_t_current
                f2 = ue.sample
                stack = [t0, t1, t2]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1, f2], axis=0)

            if config.dnobs3:

                cz_prev = cz.reuse(c_t_prev)
                cdist = self.create_component(config.u_to_c, name='z_to_u', input=random_like(cz_prev))
                t0 = x_input #self.last_frame_2
                t1 = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz_prev})
                t2 = ugb
                f0 = cz_prev 
                f1 = cz.reuse(c_t_current)
                f2 = cdist.sample
                stack = [t0, t1, t2]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1, f2], axis=0)


            if config.dnobs4:
                noise = random_like(c_t_prev)
                cdist = self.create_component(config.u_to_c, name='z_to_u', input=noise)
                ugb = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.reuse(cdist.sample)})
                t0 = x_input #self.last_frame_2
                t1 = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.reuse(c_t_prev)})
                t2 = ugb
                f0 = c_t_prev
                f1 = c_t_current
                f2 = cdist.sample
                stack = [t0, t1, t2]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1, f2], axis=0)
 
            if config.dnobs5:
                noise = random_like(c_t_prev)
                cdist = self.create_component(config.u_to_c, name='z_to_u', input=noise)
                ugb = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.reuse(cdist.sample)})
                t0 = x_input #self.last_frame_2
                t1 = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.reuse(c_t_prev)})
                f0 = c_t_prev
                f1 = c_t_current
                stack = [t0, t1]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1], axis=0)

            if config.dnobs5a:
                noise = random_like(c_t_prev)
                cdist = self.create_component(config.u_to_c, name='z_to_u', input=noise)
                ugb = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.reuse(cdist.sample)})
                t0 = self.last_frame_2
                t1 = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.reuse(c_t_current)})
                f0 = c_t_current
                f1 = c_t_prev
                stack = [t0, t1]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1], axis=0)


            if config.dnobs6:
                cz_current = cz.reuse(c_t_current)
                t0 = self.last_frame_2
                t1 = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz_current})
                t2 = ugb
                f0 = c_t_current
                f1 = c_t_prev
                f2 = ue.sample
                stack = [t0, t1, t2]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1, f2], axis=0)

            if config.dnobs7:
                noise = random_like(c_t_current)
                cz_current = cz.reuse(c_t_current)
                cz_prev = cz.reuse(c_t_prev)
                cz_fake = cz.reuse(noise)
                cdist = self.create_component(config.u_to_c, name='zu', input=noise)
                cz_fake_next = cz.reuse(C(random_like(z), cdist.sample, random=config.add_random, reuse=True).sample)
                ugb = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz_fake_next})
                t0 = self.last_frame_2
                t1 = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz_current})
                t2 = ugb
                f0 = cz_current
                f1 = cz_prev
                f2 = cz_fake
                stack = [t0, t1, t2]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1, f2], axis=0)


            if config.dnobs8:
                cz_current = cz.reuse(c_t_current)
                cz_prev = cz.reuse(c_t_prev)
                cz_fake = cz.sample
                c_t_rand = C(random_like(z), ue.sample, random=config.add_random, reuse=True).sample
                cz_fake_next = cz.reuse(c_t_rand)
                ugb = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.reuse(ue.sample)})
                t0 = self.last_frame_2
                t1 = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz_prev})
                t2 = ugb
                f0 = c_t_prev
                f1 = c_t_current
                f2 = c_t_rand
                stack = [t0, t1, t2]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1, f2], axis=0)

            if config.cnobs:
                t0 = x_input #self.last_frame_2
                t1 = ugb
                f0 = cz.reuse(c_t_current)
                f1 = cz.reuse(ue.sample)
                stack = [t0, t1]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1], axis=0)

            if config.cnobs2:
                t0 = x_input #self.last_frame_2
                t1 = ugb
                f0 = c_t_current
                f1 = ue.sample
                stack = [t0, t1]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1], axis=0)

            if config.cnobs2a:
                t0 = x_input #self.last_frame_2
                t1 = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.reuse(c_t_current)})
                f0 = c_t_current
                f1 = c_t_prev
                stack = [t0, t1]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1], axis=0)


            if config.cnobs3:
                t0 = x_input #self.last_frame_2
                t1 = ugb
                f0 = c_t_current
                f1 = ue.sample
                f2 = c_t_prev
                t2 = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.reuse(c_t_current)})
                stack = [t0, t1,t2]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1,f2], axis=0)

            if config.cnobs4:
                f1 = random_like(z)
                ugb = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":f1})
                t0 = x_input #self.last_frame_2
                t1 = ugb
                f0 = cz.reuse(c_t_current)
                f2 = cz.reuse(c_t_prev)
                t2 = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":f0})
                stack = [t0, t1,t2]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1,f2], axis=0)
 

            if config.cnobs5:
                f1 = random_like(z)
                ugb = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":f1})
                t0 = x_input #self.last_frame_2
                t1 = ugb
                f0 = cz.reuse(c_t_current)
                stack = [t0, t1]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1], axis=0)
 
            if config.nobs:
                t0 = x_input #self.last_frame_2
                t1 = ugb
                f0 = c_t_prev
                f1 = ue.sample
                stack = [t0, t1]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1], axis=0)

            if config.nobs2:
                cv0 = c_random
                ugb = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.reuse(cv0)})
                cv1 = C(generator.controls['z'], cv0, random=config.add_random, reuse=True).sample
                ugbt1 = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.reuse(cv1)})
                cv2 = C(generator.controls['z'], cv1, random=config.add_random, reuse=True).sample
                ugbt2 = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.reuse(cv2)})
                cv3 = C(generator.controls['z'], cv2, random=config.add_random, reuse=True).sample
                
                t0 = tf.concat([self.last_frame_1, self.last_frame_2, x_input], axis=3)
                t1 = tf.concat([ugb, ugbt1, ugbt2], axis=3)
                f0 = tf.concat([unrolled_c.sample, c_t_prev, c_t_current], axis=3)
                f1 = tf.concat([cv1, cv2, cv3], axis=3)
                print("TTT", t1, f1)
                stack = [t0, t1]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1], axis=0)           
            if config.nobs3:
                cv0 = c_random
                ugb = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.reuse(cv0)})
                cv1 = C(generator.controls['z'], cv0, random=config.add_random, reuse=True).sample
                ugbt1 = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.reuse(cv1)})
                cv2 = C(generator.controls['z'], cv1, random=config.add_random, reuse=True).sample
                ugbt2 = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.reuse(cv2)})
                cv3 = C(generator.controls['z'], cv2, random=config.add_random, reuse=True).sample
                
                t0 = tf.concat([self.last_frame_1, self.last_frame_2, x_input], axis=3)
                t1 = tf.concat([ugb, ugbt1, ugbt2], axis=3)
                f0 = tf.concat([unrolled_c.sample, c_t_prev, c_t_current], axis=3)
                f1 = tf.concat([cv0, cv1, cv2], axis=3)
                print("TTT", t1, f1)
                stack = [t0, t1]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1], axis=0)           

            if config.nobs4:
                cv0 = c_random
                ugb = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.reuse(cv0)})
                cv1 = C(generator.controls['z'], cv0, random=config.add_random, reuse=True).sample
                ugbt1 = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.reuse(cv1)})
                cv2 = C(generator.controls['z'], cv1, random=config.add_random, reuse=True).sample
                ugbt2 = generator.reuse(tf.zeros_like(x_input), replace_controls={"z":cz.reuse(cv2)})
                cv3 = C(generator.controls['z'], cv2, random=config.add_random, reuse=True).sample
                
                t0 = tf.concat([self.last_frame_1, self.last_frame_2, x_input], axis=3)
                t1 = tf.concat([ugb, ugbt1, ugbt2], axis=3)
                f0 = tf.concat([unrolled_c.sample, c_t_prev, c_t_current], axis=3)
                f1 = tf.concat([c_random, cv0, cv1], axis=3)
                print("TTT", t1, f1)
                stack = [t0, t1]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f0, f1], axis=0)           



            d = self.create_component(config.discriminator, name='d_b', input=stacked, features=[features])
            l = self.create_loss(config.loss, d, x_input, generator.sample, len(stack))
            loss1 = l
            d_loss1 = l.d_loss
            g_loss1 = l.g_loss

            d_vars1 = d.variables()
            g_vars1 = generator.variables() + cz.variables() + x_encoded.variables() + unrolled_c.variables()
            g_vars1 += cdist.variables()

            d_loss = l.d_loss
            g_loss = l.g_loss

            metrics = {}



            self.c_t_prev = c_t_prev
            self.c_t_current = c_t_current

            self.c_in = c_random

            metrics['lossd']=l.d_loss
            metrics['lossg']=l.g_loss

            trainers = []

            lossa = hc.Config({'sample': [d_loss1, g_loss1], 'metrics': metrics})
            #lossb = hc.Config({'sample': [d_loss2, g_loss2], 'metrics': metrics})
            trainers += [ConsensusTrainer(self, config.trainer, loss = lossa, g_vars = g_vars1, d_vars = d_vars1)]
            #trainers += [ConsensusTrainer(self, config.trainer, loss = lossb, g_vars = g_vars2, d_vars = d_vars2)]
            trainer = MultiTrainerTrainer(trainers)
            self.session.run(tf.global_variables_initializer())

        self.trainer = trainer
        self.generator = generator
        self.encoder = hc.Config({"sample":ugb}) # this is the other gan
        self.uniform_encoder = hc.Config({"sample":z})#uniform_encoder
        self.x_input = x_input
        self.x_hat = x_hat#tf.zeros_like(x_input)

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

        self.c, self.x = gan.session.run([gan.c,gan.x_input])
        self.i = 0
        BaseSampler.__init__(self, gan, samples_per_row)

    def _sample(self):
        gan = self.gan
        z_t = gan.uniform_encoder.sample
        sess = gan.session

        self.c, self.x = sess.run([gan.c_next, gan.video_next], {gan.c: self.c, gan.x_input: self.x})
        v = sess.run(gan.video_sample)
        #next_z, next_frame = sess.run([gan.cz_next, gan.video_sample])

        time.sleep(0.05)
        return {


            'generator': np.hstack([self.x, v])
        }


class TrainingVideoFrameSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        self.z = None

        self.x_input, self.last_frame_1, self.last_frame_2, self.z1, self.z2 = gan.session.run([gan.x_input, gan.inputs.x1, gan.inputs.x2, gan.z1, gan.z2])

        self.i = 0
        BaseSampler.__init__(self, gan, samples_per_row)

    def _sample(self):
        gan = self.gan
        z_t = gan.uniform_encoder.sample
        sess = gan.session
        
        x_hat,  next_frame, c = sess.run([gan.x_hat, gan.x_next, gan.c], {gan.x_input:self.x_input, gan.last_frame_1:self.last_frame_1, gan.last_frame_2:self.last_frame_2})
        xt1, c = sess.run([gan.x_next, gan.c], {gan.x_input:next_frame, gan.c2:c})
        xt2, c = sess.run([gan.x_next, gan.c], {gan.x_input:next_frame, gan.c2:c})
 
        return {
            'generator': np.vstack([self.last_frame_1, self.last_frame_2, x_hat, next_frame, xt1, xt2])
        }




def setup_gan(config, inputs, args):
    gan = AliNextFrameGAN(config, inputs=inputs)

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
