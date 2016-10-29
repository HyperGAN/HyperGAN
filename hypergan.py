import hyperchamber as hc
from lib.util.ops import *
from lib.util.globals import *
from lib.gan import *
from lib.util.gan_server import *
from tensorflow.contrib import ffmpeg
import lib.util.hc_tf as hc_tf
import lib.generators.resize_conv as resize_conv
import lib.trainers.adam_trainer as adam_trainer
import lib.trainers.rmsprop_trainer as rmsprop_trainer
import lib.trainers.slowdown_trainer as slowdown_trainer
import lib.trainers.sgd_adam_trainer as sgd_adam_trainer
import lib.discriminators.pyramid_discriminator as pyramid_discriminator
import lib.discriminators.pyramid_nostride_discriminator as pyramid_nostride_discriminator
import lib.discriminators.densenet_discriminator as densenet_discriminator
import lib.encoders.random_encoder as random_encoder
import lib.samplers.progressive_enhancement_sampler as progressive_enhancement_sampler
import lib.regularizers.minibatch_regularizer as minibatch_regularizer
import lib.regularizers.moment_regularizer as moment_regularizer
import lib.regularizers.progressive_enhancement_minibatch_regularizer as progressive_enhancement_minibatch_regularizer
import lib.regularizers.l2_regularizer as l2_regularizer
import json
import uuid
import time

import lib.loaders.image_loader
import lib.loaders.audio_loader
import os
import sys
import time
import numpy as np
import tensorflow
import tensorflow as tf
import copy

import matplotlib
import matplotlib.pyplot as plt

from tensorflow.python.framework import ops

import importlib

import lib.cli as cli

args = cli.parse_args()

#The data type to use in our GAN.  Only float32 is supported at the moment
hc.set('dtype', tf.float32)
# Generator configuration
hc.set("generator", resize_conv.generator)
hc.set("generator.z_projection_depth", 1024) # Used in the first layer - the linear projection of z
hc.set("generator.activation", [tf.nn.elu, tf.nn.relu, tf.nn.relu6, lrelu]); # activation function used inside the generator
hc.set("generator.activation.end", [tf.nn.tanh]); # Last layer of G.  Should match the range of your input - typically -1 to 1
hc.set("generator.fully_connected_layers", 0) # Experimental - This should probably stay 0
hc.set("generator.final_activation", [tf.nn.tanh]) #This should match the range of your input
hc.set("generator.resize_conv.depth_reduction", 1.5) # Divides our depth by this amount every time we go up in size
hc.set("generator.regularizers", [[]]) # These are added to the loss function for G.
hc.set("generator.regularizers.l2.lambda", list(np.linspace(0.1, 1, num=30))) # the magnitude of the l2 regularizer(experimental)

# Trainer configuration
#trainer = adam_trainer
trainer = slowdown_trainer
#trainer = sgd_adam_trainer
hc.set("trainer.initializer", trainer.initialize)
hc.set("trainer.train", trainer.train)
#Adam trainer
hc.set("trainer.adam.discriminator.lr", 1e-3) #adam_trainer d learning rate
hc.set("trainer.adam.discriminator.epsilon", 1e-8) #adam epsilon for d
hc.set("trainer.adam.discriminator.beta1", 0.9) #adam epsilon for d
hc.set("trainer.adam.discriminator.beta2", 0.999) #adam epsilon for d
hc.set("trainer.adam.generator.lr", 1e-3) #adam_trainer g learning rate
hc.set("trainer.adam.generator.epsilon", 1e-8) #adam_trainer g learning rate
hc.set("trainer.adam.generator.beta1", 0.9) #adam_trainer g learning rate
hc.set("trainer.adam.generator.beta2", 0.999) #adam_trainer g learning rate
#This trainer slows D down when d_fake gets too high
hc.set("trainer.rmsprop.discriminator.lr", 1.5e-5) # d learning rate
hc.set('trainer.slowdown.discriminator.d_fake_min', [0.12]) # healthy above this number on d_fake
hc.set('trainer.slowdown.discriminator.d_fake_max', [0.12001]) # unhealthy below this number on d_fake
hc.set('trainer.slowdown.discriminator.slowdown', [5]) # Divides speed by this number when unhealthy(d_fake low)
#This trainer uses SGD on D and adam on G
hc.set("trainer.sgd_adam.discriminator.lr", 1e-2) # d learning rate
hc.set("trainer.sgd_adam.generator.lr", 1e-3) # g learning rate

# Discriminator configuration
hc.set("discriminator", pyramid_nostride_discriminator.discriminator)
hc.set("discriminator.activation", [tf.nn.elu, tf.nn.relu, tf.nn.relu6, lrelu]);

hc.set('discriminator.fc_layer', [False]) #If true, include a fully connected layer at the end of the discriminator
hc.set('discriminator.fc_layers', [1])# Number of fully connected layers to include
hc.set('discriminator.fc_layer.size', 512) # Size of fully connected layers

hc.set("discriminator.pyramid.layers", 6) #Layers in D
hc.set("discriminator.pyramid.depth_increase", 2)# Size increase of D's features on each layer

hc.set('discriminator.densenet.k', 16) #k is the number of features that are appended on each conv pass
hc.set('discriminator.densenet.layers', 2) #number of times to conv before size transition
hc.set('discriminator.densenet.transitions', 6) #number of transitions

hc.set('discriminator.add_noise', [True]) #add noise to input
hc.set('discriminator.noise_stddev', [1e-1]) #the amount of noise to add - always centered at 0
hc.set('discriminator.regularizers', [[]]) # these regularizers get applied at the end of D

hc.set("sampler", progressive_enhancement_sampler.sample)
hc.set("sampler.samples", 3)
hc.set('encoder.sample', random_encoder.sample) # how to encode z

#TODO vae
#hc.set("transfer_fct", [tf.nn.elu, tf.nn.relu, tf.nn.relu6, lrelu]);
#hc.set("e_activation", [tf.nn.elu, tf.nn.relu, tf.nn.relu6, lrelu]);
#hc.set("e_last_layer", [tf.nn.tanh]);
#hc.set('f_skip_fc', False)
#hc.set('f_hidden_1', 512)#list(np.arange(256, 512)))
#hc.set('f_hidden_2', 256)#list(np.arange(256, 512)))

## Below here are legacy settings that need to be cleaned up - they may still be in use
#TODO preprocess loader
#hc.set('pretrained_model', [None])

#TODO audio
#hc.set("g_mp3_dilations",[[1,2,4,8,16,32,64,128,256]])
#hc.set("g_mp3_filter",[3])
#hc.set("g_mp3_residual_channels", [8])
#hc.set("g_mp3_dilation_channels", [16])
#hc.set("mp3_seconds", args.seconds)
#hc.set("mp3_bitrate", args.bitrate)
#hc.set("mp3_size", args.seconds*args.bitrate)

hc.set("model", "faces:1.0")

hc.set("z_dim", 64) #TODO rename to generator.z

#TODO category/bernouilli
categories = [[2]+[2]+build_categories_config(30)]
hc.set('categories', categories)
hc.set('categories_lambda', list(np.linspace(.001, .01, num=100)))
hc.set('category_loss', [False])

#TODO loss functions
hc.set('g_class_loss', [False])
hc.set('g_class_lambda', list(np.linspace(0.01, .1, num=30)))
hc.set('d_fake_class_loss', [False])

#TODO one-sided label smoothing loss
hc.set("g_target_prob", list(np.linspace(.65 /2., .85 /2., num=100)))
hc.set("d_label_smooth", list(np.linspace(0.15, 0.35, num=100)))

#TODO move to minibatch
hc.set("d_kernels", list(np.arange(20, 30)))
hc.set("d_kernel_dims", list(np.arange(100, 300)))

#TODO remove and replace with losses
hc.set("loss", ['custom'])

#TODO Vae Loss
hc.set("adv_loss", [False])
hc.set("latent_loss", [False])
hc.set("latent_lambda", list(np.linspace(.01, .1, num=30)))

#TODO Is this about adding z to the D?  Is this right? Investigate
hc.set("d_project", ['tiled'])

hc.set("batch_size", args.batch)
hc.set("format", args.format)


def epoch(sess, config):
    batch_size = config["batch_size"]
    n_samples =  config['examples_per_epoch']
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        d_loss, g_loss = config['trainer.train'](sess, config)
        if(i > 10 and not args.no_stop):
            if(math.isnan(d_loss) or math.isnan(g_loss) or g_loss > 1000 or d_loss > 1000):
                return False

            g = get_tensor('g')
            rX = sess.run([g[-1]])
            rX = np.array(rX)
            if(np.min(rX) < -1000 or np.max(rX) > 1000):
                return False
    return True

def test_config(sess, config):
    batch_size = config["batch_size"]
    n_samples =  batch_size*10
    total_batch = int(n_samples / batch_size)
    results = []
    for i in range(total_batch):
        results.append(test(sess, config))
    return results

def collect_measurements(epoch, sess, config, time):
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

def test_epoch(epoch, sess, config, start_time, end_time):
    sample = []
    sample_list = config['sampler'](sess,config)
    measurements = collect_measurements(epoch, sess, config, end_time - start_time)
    hc.io.measure(config, measurements)
    hc.io.sample(config, sample_list)


def get_function(name):
    if not isinstance(name, str):
        return name
    namespaced_method = name.split(":")[1]
    method = namespaced_method.split(".")[-1]
    namespace = ".".join(namespaced_method.split(".")[0:-1])
    return getattr(importlib.import_module(namespace),method)

# Take a config and replace any string starting with 'function:' with a function lookup.
def lookup_functions(config):
    for key, value in config.items():
        if(isinstance(value, str) and value.startswith("function:")):
            config[key]=get_function(value)
        if(isinstance(value, list) and len(value) > 0 and isinstance(value[0],str) and value[0].startswith("function:")):
            config[key]=[get_function(v) for v in value]
            
    return config


def run(args):
    print("Generating configs with hyper search space of ", hc.count_configs())
    for config in hc.configs(1):
        other_config = copy.copy(config)
        if(args.load_config):
            print("Loading config", args.load_config)
            config.update(hc.io.load_config(args.load_config))
            if(not config):
                print("Could not find config", args.load_config)
                break

        config = lookup_functions(config)
        config['batch_size']=args.batch
        config['dtype']=other_config['dtype']
        with tf.device(args.device):
            sess = tf.Session(config=tf.ConfigProto())
        channels = args.channels
        crop = args.crop
        width = args.width
        height = args.height
        with tf.device('/cpu:0'):
            if(args.format == 'mp3'):
                train_x,train_y, num_labels,examples_per_epoch = lib.loaders.audio_loader.mp3_tensors_from_directory(args.directory,config['batch_size'], seconds=args.seconds, channels=channels, bitrate=args.bitrate, format=args.format)
                f = None
            else:
                train_x,train_y, f, num_labels,examples_per_epoch = lib.loaders.image_loader.labelled_image_tensors_from_directory(args.directory,config['batch_size'], channels=channels, format=args.format,crop=crop,width=width,height=height)
        config['y_dims']=num_labels
        config['x_dims']=[height,width]
        config['channels']=channels

        if(args.load_config):
            pass
        config['examples_per_epoch']=examples_per_epoch//4
        x = train_x
        y = train_y
        with tf.device(args.device):
            y=tf.one_hot(tf.cast(train_y,tf.int64), config['y_dims'], 1.0, 0.0)

            if(args.build or args.server):
                graph = create_generator(config,x,y,f)
            else:
                graph = create(config,x,y,f)
        saver = tf.train.Saver()
        if('parent_uuid' in config):
            save_file = "saves/"+config["parent_uuid"]+".ckpt"
        else:
            save_file = "saves/"+config["uuid"]+".ckpt"
        if(save_file and os.path.isfile(save_file) and not args.server):
            print(" |= Loading network from "+ save_file)
            config['uuid']=config['parent_uuid']
            ckpt = tf.train.get_checkpoint_state('saves')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, save_file)
                print("Model loaded")
            else:
                print("No checkpoint file found")
        else:
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
            print("Top 5 sizes", sizes[-5:])
            size = sum([s[1] for s in sizes])
            print("SIZE = ", size)

            with tf.device(args.device):
                init = tf.initialize_all_variables()
                sess.run(init)

        tf.train.start_queue_runners(sess=sess)
        testx = sess.run(train_x)

        build_file = "build/generator.ckpt"
        if args.build:
            saver.save(sess, build_file)
            print("Saved generator to ", build_file)
        elif args.server:
            print("Loading from build/")
            saver.restore(sess, build_file)
            gan_server(sess, config)
        else:
            sampled=False
            print("Running for ", args.epochs, " epochs")
            for i in range(args.epochs):
                start_time = time.time()
                with tf.device(args.device):
                    if(not epoch(sess, config)):
                        print("Epoch failed")
                        break
                print("Checking save "+ str(i))
                if(args.save_every != 0 and i % args.save_every == args.save_every-1):
                    print(" |= Saving network")
                    saver.save(sess, save_file)
                end_time = time.time()
                test_epoch(i, sess, config, start_time, end_time)

            tf.reset_default_graph()
            sess.close()

if __name__ == "__main__":
    run(args) 
    

