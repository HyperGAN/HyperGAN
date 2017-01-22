from hypergan.util.gan_server import *
from hypergan.util.globals import *
from hypergan.util.ops import *

from tensorflow.python.framework import ops

import copy

import hyperchamber as hc

import hypergan as hg

from hypergan.discriminators import *
from hypergan.encoders import *
from hypergan.generators import *
from hypergan.loaders import *
from hypergan.regularizers import *
from hypergan.samplers import *
from hypergan.trainers import *
from hypergan.util import *

import importlib
import json
import numpy as np
import os
import sys
import tensorflow
import tensorflow as tf
import time
import uuid

batch_no = 0
sampled = 0

class GAN:
    """ GANs (Generative Adversarial Networks) consist of generator(s) and discriminator(s)."""
    def __init__(self, config, device='/gpu:0'):
        """ Initialized a new GAN."""
        self.config=config
        self.init_session(device)
        if(args.method == 'build' or args.method == 'serve'):
            graph_type = 'generator'
        else:
            graph_type = 'full'

        self.graph = self.create_graph(x, y, f, graph_type, device)


    def sample_file(self, name, sampler=grid_sampler):
        sampler.sample(name, self.sess, self.config)

    # This looks up a function by name.   Should it be part of hyperchamber?
    #TODO moveme
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
    #TODO moveme
    def lookup_functions(self, config):
        for key, value in config.items():
            if(isinstance(value, str) and value.startswith("function:")):
                config[key]=self.get_function(value)
            if(isinstance(value, list) and len(value) > 0 and isinstance(value[0],str) and value[0].startswith("function:")):
                config[key]=[self.get_function(v) for v in value]

        return config

    def create_graph(self, x, y, f, graph_type, device):
        self.graph = hg.graph.Graph(self.config)

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


    def init_session(self, device):
        # Initialize tensorflow
        with tf.device(device):
            self.sess = tf.Session(config=tf.ConfigProto())

    def run(self):
        print( "Save file", save_file,"\n")
        #TODO refactor save/load system
        if args.method == 'serve':
            print("|= Loading generator from build/")
            saver = tf.train.Saver()
            saver.restore(self.sess, build_file)
        elif(save_file and ( os.path.isfile(save_file) or os.path.isfile(save_file + ".index" ))):
            print(" |= Loading network from "+ save_file)
            ckpt = tf.train.get_checkpoint_state(os.path.expanduser('~/.hypergan/saves/'))
            if ckpt and ckpt.model_checkpoint_path:
                saver = tf.train.Saver()
                saver.restore(self.sess, save_file)
                loadedFromSave = True
                print("Model loaded")
            else:
                print("No checkpoint file found")
        else:
            print(" |= Initializing new network")
            with tf.device(args.device):
                init = tf.initialize_all_variables()
                self.sess.run(init)

        self.output_graph_size()
        tf.train.start_queue_runners(sess=self.sess)
        testx = self.sess.run(x)

        if args.method == 'build':
            saver = tf.train.Saver()
            saver.save(self.sess, build_file)
            print("Saved generator to ", build_file)
        elif args.method == 'serve':
            gan_server(self.sess, config)
        else:
            sampled=False
            print("Running for ", args.epochs, " epochs")
            for i in range(args.epochs):
                start_time = time.time()
                with tf.device(args.device):
                    if(not self.epoch(self.sess, config)):
                        print("Epoch failed")
                        break
                print("Checking save "+ str(i))
                if(args.save_every != 0 and i % args.save_every == args.save_every-1):
                    print(" |= Saving network")
                    saver = tf.train.Saver()
                    saver.save(self.sess, save_file)
                end_time = time.time()
                self.test_epoch(i, self.sess, config, start_time, end_time)

            tf.reset_default_graph()
            self.sess.close()
