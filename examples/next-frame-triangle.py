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

        x  = tf.train.slice_input_producer([filenames], shuffle=True)[0]
        y  = tf.train.slice_input_producer([filenames], shuffle=True)[0]
        self.x = self.read_frame(x, format, crop, resize)
        self.y = self.read_frame(y, format, crop, resize)
        self.x = self._get_iid_data(self.x)
        self.y = self._get_iid_data(self.y)
        return [x1, x2, x3, self.x, self.y], None


    def _get_iid_data(self, imgs):
        batch_size = self.batch_size
        num_preprocess_threads = 24
        xs = tf.train.shuffle_batch(
            [imgs],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity= batch_size*10,
            min_after_dequeue=batch_size)
        return xs

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
            def random_like(x):
                return UniformEncoder(self, config.z_distribution, output_shape=self.ops.shape(x)).sample
            #x_input = tf.identity(self.inputs.x, name='input')
            x_input = tf.identity(self.inputs.x3, name='xb_i')
            last_frame_1 = tf.identity(self.inputs.x1, name='x1_t')
            last_frame_2 = tf.identity(self.inputs.x2, name='x2_t')
            self.last_frame_1 = last_frame_1
            self.last_frame_2 = last_frame_2


            xa_input = self.inputs.y
            xb_input = self.inputs.x
            if config.seq:
                xa_input = last_frame_1
                xb_input = last_frame_2
            if config.seq2:
                xa_input = last_frame_2
                xb_input = x_input

            zgx = self.create_component(config.encoder, input=xa_input, name='xa_to_x')
            zgy = self.create_component(config.encoder, input=xb_input, name='xb_to_y')
            zx = zgx.sample
            zy = zgy.sample
            z_noise = random_like(zx)
            n_noise = random_like(zx)
            zx_noise = z_noise
            zy_noise = z_noise
            
            if config.unique_noise:
                zy_noise = random_like(z_noise)

            if config.style:
                stylex = self.create_component(config.style_discriminator, input=xb_input, name='xb_style')
                styley = self.create_component(config.style_discriminator, input=xa_input, name='xa_style')
                zy = tf.concat(values=[zy, z_noise], axis=3)
                gy = self.create_component(config.generator, features=[styley.sample], input=zy, name='gy_generator')
                y = hc.Config({"sample": xa_input})
                zx2 = self.create_component(config.encoder, input=y.sample, name='xa_to_x', reuse=True).sample
                if config.add_last_frame:
                    zx2prime = self.create_component(config.encoder, input=xa_input, name='xb_to_y', reuse=True)
                    zx2 = tf.concat(values=[zx2, zx2prime.sample, z_noise], axis=3)
                else:
                    zx2 = tf.concat(values=[zx2, z_noise], axis=3)
                gx = self.create_component(config.generator, features=[stylex.sample], input=zx2, name='gx_generator')
            else:
                gy = self.create_component(config.generator, features=[zy_noise], input=zy, name='gy_generator')
                y = hc.Config({"sample": xa_input})
                zx2 = self.create_component(config.encoder, input=y.sample, name='xa_to_x', reuse=True).sample
                gx = self.create_component(config.generator, features=[zx_noise], input=zx2, name='gx_generator')
                stylex=hc.Config({"sample":random_like(y.sample)})

            self.y = y
            self.gy = gy
            self.gx = gx

            ga = gy
            gb = gx

            self.uniform_sample = gb.sample

            xba = ga.sample
            xab = gb.sample
            xa = xa_input
            xb = xb_input

            self.styleb = stylex
            self.random_style = random_like(stylex.sample)


            t0 = self.last_frame_2
            f0 = self.last_frame_1
            t1 = xb
            t2 = gx.sample
            f1 = gy.sample
            f2 = y.sample
            stack = [t0, t1, t2]
            stacked = ops.concat(stack, axis=0)
            features = ops.concat([f0, f1, f2], axis=0)

            if config.d2_only:
                t1 = xb
                t2 = gx.sample
                f1 = gy.sample
                f2 = y.sample
                stack = [t1, t2]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f1, f2], axis=0)



            if config.skip_real:
                stack = [t1, t2]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f1, f2], axis=0)


            self.inputs.x = xa
            zub = zy
            sourcezub = zy


            d = self.create_component(config.discriminator, name='d_ab', 
                    input=stacked, features=[features])
            if config.mask_d:
                stack = [t1, t2]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f1, f2], axis=0)
                d2 = self.create_component(config.discriminator, name='d2', 
                        input=stacked, features=[features])
                real2, fake2 = self.split_batch(d2.sample)
                if config.skip_real:
                    real, fake = self.split_batch(d.sample, count=2)
                    real = (1.-tf.nn.sigmoid(real))*real2
                    fake = (1.-tf.nn.sigmoid(fake))*fake2
                    #real *= real2
                    #fake *= (1-fake2)
                    d.sample = tf.concat([real,fake], axis=0)
                else:

                    keep, real, fake = self.split_batch(d.sample, count=3)
                    real = (1.-tf.nn.sigmoid(real))*real2
                    fake = (1.-tf.nn.sigmoid(fake))*fake2
                    #real *= real2
                    #fake *= (1-fake2)
                    d.sample = tf.concat([keep,real,fake], axis=0)
            if config.mask_d2:
                stack = [t1, t2]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f1, f2], axis=0)
                d2 = self.create_component(config.discriminator, name='d2', 
                        input=stacked, features=[features])
                real2, fake2 = self.split_batch(d2.sample)
                keep, real, fake = self.split_batch(d.sample, count=3)
                real = (1.-real)*real2
                fake = (1.-fake)*(1.-fake2)
                #real *= real2
                #fake *= (1-fake2)
                d.sample = tf.concat([keep,real,fake], axis=0)


                
            l = self.create_loss(config.loss, d, xa_input, ga.sample, len(stack))
            loss1 = l
            d_loss1 = l.d_loss
            g_loss1 = l.g_loss

            d_vars1 = d.variables()
            if config.mask_d or config.mask_d2:
                d_vars1 += d2.variables()
            g_vars1 = gb.variables()+ga.variables()+zgx.variables()+zgy.variables()
            if config.fancy_y:
                g_vars1 += lfzg.variables() + y.variables()
            d_loss = l.d_loss
            g_loss = l.g_loss
            metrics = {
                    'g_loss': l.g_loss,
                    'd_loss': l.d_loss
                }

            if config.alice:
                if config.style:
                    xb_hat_z = self.create_component(config.encoder, input=gy.sample, name='xa_to_x', reuse=True).sample
                    if config.add_last_frame:
                        zyy = self.create_component(config.encoder, input=gy.sample, name='xb_to_y', reuse=True)
                        xb_hat_z = tf.concat([xb_hat_z, zyy.sample, zx_noise], axis=3)
                    else:
                        xb_hat_z = tf.concat([xb_hat_z, zx_noise], axis=3)
                    xb_hat = self.create_component(config.generator, features=[stylex.sample], input=xb_hat_z, name='gx_generator', reuse=True)
                else:
                    xb_hat_z = self.create_component(config.encoder, input=gy.sample, name='xa_to_x', reuse=True).sample
                    xb_hat = self.create_component(config.generator, features=[zx_noise], input=xb_hat_z, name='gx_generator', reuse=True)

                t1 = xb
                t2 = xb_hat.sample
                f1 = xb
                f2 = xb
                stack = [t1, t2]
                stacked = ops.concat(stack, axis=0)
                features = ops.concat([f1, f2], axis=0)
                z_d = self.create_component(config.discriminator, name='alice_discriminator', input=stacked, features=[features])
                loss3 = self.create_component(config.loss, discriminator = z_d, x=xa_input, generator=ga, split=2)
                metrics["alice_gloss"]=loss3.g_loss
                metrics["alice_dloss"]=loss3.d_loss
                d_loss1 += loss3.d_loss
                g_loss1 += loss3.g_loss
                d_vars1 += z_d.variables()



            if config.alpha:
                t0 = random_like(zx)
                t1 = zx
                t2 = zy
                netzd = tf.concat(axis=0, values=[t0,t1,t2])
                z_d = self.create_component(config.z_discriminator, name='z_discriminator', input=netzd)
                loss3 = self.create_component(config.loss, discriminator = z_d, x=xa_input, generator=ga, split=3)
                metrics["za_gloss"]=loss3.g_loss
                metrics["za_dloss"]=loss3.d_loss
                d_loss1 += loss3.d_loss
                g_loss1 += loss3.g_loss
                d_vars1 += z_d.variables()

            trainers = []

            lossa = hc.Config({'sample': [d_loss1, g_loss1], 'metrics': metrics})
            #lossb = hc.Config({'sample': [d_loss2, g_loss2], 'metrics': metrics})
            trainers += [ConsensusTrainer(self, config.trainer, loss = lossa, g_vars = g_vars1, d_vars = d_vars1)]
            #trainers += [ConsensusTrainer(self, config.trainer, loss = lossb, g_vars = g_vars2, d_vars = d_vars2)]
            trainer = MultiTrainerTrainer(trainers)
            self.session.run(tf.global_variables_initializer())

        self.trainer = trainer
        self.generator = gb
        self.uniform_encoder = hc.Config({"sample":zub})#uniform_encoder
        self.uniform_encoder_source = hc.Config({"sample":sourcezub})#uniform_encoder
        self.zb = zy
        self.z_hat = gb.sample
        self.x_input = xa_input

        self.xba = xba
        self.xab = xab
        self.uga = y.sample



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
    GlobalViewer.zoom = 2

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
