from hypergan.util.gan_server import *
from hypergan.util.globals import *
from hypergan.util.ops import *

from tensorflow.python.framework import ops

import copy

import hyperchamber as hc

import hypergan.config
import hypergan.discriminators.densenet_discriminator as densenet_discriminator
import hypergan.discriminators.fast_densenet_discriminator as fast_densenet_discriminator
import hypergan.discriminators.painters_discriminator as painters_discriminator
import hypergan.discriminators.pyramid_discriminator as pyramid_discriminator
import hypergan.discriminators.pyramid_nostride_discriminator as pyramid_nostride_discriminator
import hypergan.discriminators.slim_stride as slim_stride
import hypergan.encoders.progressive_variational_encoder as progressive_variational_encoder
import hypergan.encoders.random_combo_encoder as random_combo_encoder
import hypergan.encoders.random_encoder as random_encoder
import hypergan.encoders.random_gaussian_encoder as random_gaussian_encoder
import hypergan.generators.dense_resize_conv as dense_resize_conv
import hypergan.generators.resize_conv as resize_conv
import hypergan.generators.resize_conv_extra_layer as resize_conv_extra_layer
import hypergan.loaders.audio_loader as audio_loader
import hypergan.loaders.image_loader as image_loader
import hypergan.regularizers.l2_regularizer as l2_regularizer
import hypergan.regularizers.minibatch_regularizer as minibatch_regularizer
import hypergan.regularizers.moment_regularizer as moment_regularizer
import hypergan.regularizers.progressive_enhancement_minibatch_regularizer as progressive_enhancement_minibatch_regularizer
import hypergan.samplers.grid_sampler as grid_sampler
import hypergan.samplers.progressive_enhancement_sampler as progressive_enhancement_sampler
import hypergan.trainers.adam_trainer as adam_trainer
import hypergan.trainers.rmsprop_trainer as rmsprop_trainer
import hypergan.trainers.sgd_adam_trainer as sgd_adam_trainer
import hypergan.trainers.slowdown_trainer as slowdown_trainer
import hypergan.util.hc_tf as hc_tf

import importlib
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow
import tensorflow as tf
import time
import time
import uuid

batch_no = 0
sampled = 0

from IPython import embed

class GAN:
    """ GANs (Generative Adversarial Networks) consist of generator(s) and discriminator(s)."""
    def __init__(self, config={}, args={}):
        """ Initialized a new GAN.  Any options not specified will be randomly selected. """
        self.args = args
        self.selector = hypergan.config.selector(args)
        self.config = self.selector.random_config()
        self.config.update(config)
        # TODO load / save config?

    def frame_sample(self, sample_file, sess, config):
        """ Samples every frame to a file.  Useful for visualizing the learning process.

        Use with:

             ffmpeg -i samples/grid-%06d.png -vcodec libx264 -crf 22 -threads 0 grid1-7.mp4

        to create a video of the learning process.
        """

        if(args.frame_sample == None):
            return None
        if(args.frame_sample == "grid"):
            frame_sampler = grid_sampler.sample
        else:
            raise "Cannot find frame sampler: '"+args.frame_sample+"'"

        frame_sampler(sample_file, sess, config)

    def epoch(self, sess, config):
        batch_size = config["batch_size"]
        n_samples =  config['examples_per_epoch']
        total_batch = int(n_samples / batch_size)
        global sampled
        global batch_no
        for i in range(total_batch):
            if(i % 10 == 1):
                sample_file="samples/grid-%06d.png" % (sampled)
                self.frame_sample(sample_file, sess, config)
                sampled += 1


            d_loss, g_loss = config['trainer.train'](sess, config)

            #if(i > 10):
            #    if(math.isnan(d_loss) or math.isnan(g_loss) or g_loss > 1000 or d_loss > 1000):
            #        return False

            #    g = get_tensor('g')
            #    rX = sess.run([g[-1]])
            #    rX = np.array(rX)
            #    if(np.min(rX) < -1000 or np.max(rX) > 1000):
            #        return False
        batch_no+=1
        return True

    def test_config(self, sess, config):
        batch_size = config["batch_size"]
        n_samples =  batch_size*10
        total_batch = int(n_samples / batch_size)
        results = []
        for i in range(total_batch):
            results.append(test(sess, config))
        return results

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

    def test_epoch(self, epoch, sess, config, start_time, end_time):
        sample = []
        sample_list = config['sampler'](sess,config)
        measurements = self.collect_measurements(epoch, sess, config, end_time - start_time)
        args = cli.parse_args()
        if args.use_hc_io:
            hc.io.measure(config, measurements)
            hc.io.sample(config, sample_list)
        else:
            print("Offline sample created:", sample_list)


    # This looks up a function by name.   Should it be part of hyperchamber?
    def get_function(self, name):
        if name == "function:hypergan.util.ops.prelu_internal":
            return prelu("g_")

        if not isinstance(name, str):
            return name
        namespaced_method = name.split(":")[1]
        method = namespaced_method.split(".")[-1]
        namespace = ".".join(namespaced_method.split(".")[0:-1])
        return getattr(importlib.import_module(namespace),method)

    # Take a config and replace any string starting with 'function:' with a function lookup.
    def lookup_functions(self, config):
        for key, value in config.items():
            if(isinstance(value, str) and value.startswith("function:")):
                config[key]=self.get_function(value)
            if(isinstance(value, list) and len(value) > 0 and isinstance(value[0],str) and value[0].startswith("function:")):
                config[key]=[self.get_function(v) for v in value]

        return config


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

    def load_config(self, name):
        config = self.config
        if config is not None:
            other_config = copy.copy(dict(self.config))
            # load_saved_checkpoint(config)
            print("[hypergan] Creating or loading configuration in ~/.hypergan/configs/", name)

            config_path = os.path.expanduser('~/.hypergan/configs/'+name+'.json')
            print("Loading "+config_path)
            config = self.selector.load_or_create_config(config_path, config)

        config = self.lookup_functions(config)
        self.config = config
        return self.config

    def create_graph(self, x, y, f, graph_type, device):
        self.graph = hypergan.graph.Graph(self.config)

        with tf.device(device):
            y=tf.cast(y,tf.int64)
            y=tf.one_hot(y, self.config['y_dims'], 1.0, 0.0)

            if graph_type == 'full':
                graph = self.graph.create(x,y,f)
            elif graph_type == 'generator':
                graph = self.graph.create_generator(x,y,f)
            else:
                raise Exception("Invalid graph type")

        return self.graph

    def setup_loader(self, format, directory, device, seconds=None,
            bitrate=None, crop=False, width=None, height=None, channels=3):
        with tf.device('/cpu:0'):
            #TODO mp3 braken
            if(format == 'mp3'):
                return audio_loader.mp3_tensors_from_directory(
                        directory,
                        self.config['batch_size'],
                        seconds=seconds,
                        channels=channels,
                        bitrate=bitrate,
                        format=format)
            else:
                return image_loader.labelled_image_tensors_from_directory(
                        directory,
                        self.config['batch_size'], 
                        channels=channels, 
                        format=format,
                        crop=crop,
                        width=width,
                        height=height)


    def run(self):
        args = self.args
        crop = args.crop
        channels = int(args.size.split("x")[2])
        width = int(args.size.split("x")[0])
        height = int(args.size.split("x")[1])
        loadedFromSave = False

        print("[hypergan] Welcome back.  You are one of ", self.selector.count_configs(), " possible configurations.")
        self.load_config(args.config)
        self.config = self.selector.random_config()

        # Initialize tensorflow
        with tf.device(args.device):
            sess = tf.Session(config=tf.ConfigProto())

        x,y,f,num_labels,examples_per_epoch = self.setup_loader(
                args.format,
                args.directory,
                args.device,
                seconds=None,
                bitrate=None,
                width=width,
                height=height,
                channels=channels,
                crop=crop
        )
        self.config['y_dims']=num_labels
        self.config['x_dims']=[height,width] #todo can we remove this?
        self.config['channels']=channels
        self.config['batch_size']=args.batch_size
        config = self.config

        if args.config is None:
            filename = '~/.hypergan/configs/'+config['uuid']+'.json'
            print("[hypergan] saving network configuration to: " + filename)
            config = self.selector.load_or_create_config(filename, config)
        else:
            save_file = "~/.hypergan/saves/"+args.config+".ckpt"
            config['uuid'] = args.config

        save_file = "~/.hypergan/saves/"+config["uuid"]+".ckpt"
        samples_path = "~/.hypergan/samples/"+config['uuid']
        save_file = os.path.expanduser(save_file)
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        os.makedirs(os.path.expanduser(samples_path), exist_ok=True)
        build_file = os.path.expanduser("~/.hypergan/builds/"+config['uuid']+"/generator.ckpt")
        os.makedirs(os.path.dirname(build_file), exist_ok=True)

        if(args.method == 'build' or args.method == 'serve'):
            graph_type = 'generator'
        else:
            graph_type = 'full'

        graph = self.create_graph(x, y, f, graph_type, args.device)

        print( "Save file", save_file,"\n")
        #TODO refactor save/load system
        if args.method == 'serve':
            print("|= Loading generator from build/")
            saver = tf.train.Saver()
            saver.restore(sess, build_file)
        elif(save_file and ( os.path.isfile(save_file) or os.path.isfile(save_file + ".index" ))):
            print(" |= Loading network from "+ save_file)
            ckpt = tf.train.get_checkpoint_state(os.path.expanduser('~/.hypergan/saves/'))
            if ckpt and ckpt.model_checkpoint_path:
                saver = tf.train.Saver()
                saver.restore(sess, save_file)
                loadedFromSave = True
                print("Model loaded")
            else:
                print("No checkpoint file found")
        else:
            print(" |= Initializing new network")
            with tf.device(args.device):
                init = tf.initialize_all_variables()
                sess.run(init)

        self.output_graph_size()
        tf.train.start_queue_runners(sess=sess)
        testx = sess.run(x)

        if args.method == 'build':
            saver = tf.train.Saver()
            saver.save(sess, build_file)
            print("Saved generator to ", build_file)
        elif args.method == 'serve':
            gan_server(sess, config)
        else:
            sampled=False
            print("Running for ", args.epochs, " epochs")
            for i in range(args.epochs):
                start_time = time.time()
                with tf.device(args.device):
                    if(not self.epoch(sess, config)):
                        print("Epoch failed")
                        break
                print("Checking save "+ str(i))
                if(args.save_every != 0 and i % args.save_every == args.save_every-1):
                    print(" |= Saving network")
                    saver = tf.train.Saver()
                    saver.save(sess, save_file)
                end_time = time.time()
                self.test_epoch(i, sess, config, start_time, end_time)

            tf.reset_default_graph()
            sess.close()
