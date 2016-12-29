import hyperchamber as hc
from hypergan.util.ops import *
from hypergan.util.globals import *
from hypergan.gan import *
from hypergan.util.gan_server import *
from tensorflow.contrib import ffmpeg
import hypergan.util.hc_tf as hc_tf
import hypergan.generators.resize_conv as resize_conv
import hypergan.generators.resize_conv_extra_layer as resize_conv_extra_layer
import hypergan.trainers.adam_trainer as adam_trainer
import hypergan.trainers.rmsprop_trainer as rmsprop_trainer
import hypergan.trainers.slowdown_trainer as slowdown_trainer
import hypergan.trainers.sgd_adam_trainer as sgd_adam_trainer
import hypergan.discriminators.pyramid_discriminator as pyramid_discriminator
import hypergan.discriminators.pyramid_nostride_discriminator as pyramid_nostride_discriminator
import hypergan.discriminators.slim_stride as slim_stride
import hypergan.discriminators.densenet_discriminator as densenet_discriminator
import hypergan.discriminators.fast_densenet_discriminator as fast_densenet_discriminator
import hypergan.discriminators.painters_discriminator as painters_discriminator
import hypergan.encoders.random_encoder as random_encoder
import hypergan.encoders.random_gaussian_encoder as random_gaussian_encoder
import hypergan.encoders.random_combo_encoder as random_combo_encoder
import hypergan.encoders.progressive_variational_encoder as progressive_variational_encoder
import hypergan.samplers.progressive_enhancement_sampler as progressive_enhancement_sampler
import hypergan.samplers.grid_sampler as grid_sampler
import hypergan.regularizers.minibatch_regularizer as minibatch_regularizer
import hypergan.regularizers.moment_regularizer as moment_regularizer
import hypergan.regularizers.progressive_enhancement_minibatch_regularizer as progressive_enhancement_minibatch_regularizer
import hypergan.regularizers.l2_regularizer as l2_regularizer
import json
import uuid
import time

import hypergan.loaders.image_loader
import hypergan.loaders.audio_loader
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

import hypergan.cli as cli

args = cli.parse_args()

# Below are sets of configuration options:
# Each time a new random network is started a random set of configuration variables are selected.
# This is useful for hyperparameter search.  If you want to use a specific configuration use --config

hc.set('dtype', tf.float32) #The data type to use in our GAN.  Only float32 is supported at the moment

# Generator configuration
hc.set("generator.z", 100) # the size of the encoding.  Encoder is set by the 'encoder' property, but could just be a random_uniform
hc.set("generator", [resize_conv.generator])
hc.set("generator.z_projection_depth", 1024) # Used in the first layer - the linear projection of z
hc.set("generator.activation", [prelu("g_")]); # activation function used inside the generator
hc.set("generator.activation.end", [tf.nn.tanh]); # Last layer of G.  Should match the range of your input - typically -1 to 1
hc.set("generator.fully_connected_layers", 0) # Experimental - This should probably stay 0
hc.set("generator.final_activation", [tf.nn.tanh]) #This should match the range of your input
hc.set("generator.resize_conv.depth_reduction", 2) # Divides our depth by this amount every time we go up in size
hc.set("generator.regularizers", [[l2_regularizer.get]]) # These are added to the loss function for G.
hc.set('generator.layer.noise', False) #Adds incremental noise each layer
hc.set("generator.regularizers.l2.lambda", list(np.linspace(0.1, 1, num=30))) # the magnitude of the l2 regularizer(experimental)
hc.set("generator.regularizers.layer", [batch_norm_1]) # the magnitude of the l2 regularizer(experimental)

# Trainer configuration
trainer = adam_trainer # adam works well at 64x64 but doesn't scale
#trainer = slowdown_trainer # this works at higher resolutions, but is slow and quirky(help wanted)
#trainer = rmsprop_trainer # this works at higher resolutions, but is slow and quirky(help wanted)
#trainer = sgd_adam_trainer # This has never worked, but seems like it should
hc.set("trainer.initializer", trainer.initialize) # TODO: can we merge these variables?
hc.set("trainer.train", trainer.train) # The training method to use.  This is called every step
hc.set("trainer.adam.discriminator.lr", 1e-3) #adam_trainer d learning rate
hc.set("trainer.adam.discriminator.epsilon", 1e-8) #adam epsilon for d
hc.set("trainer.adam.discriminator.beta1", 0.9) #adam beta1 for d
hc.set("trainer.adam.discriminator.beta2", 0.999) #adam beta2 for d
hc.set("trainer.adam.generator.lr", 1e-3) #adam_trainer g learning rate
hc.set("trainer.adam.generator.epsilon", 1e-8) #adam_trainer g
hc.set("trainer.adam.generator.beta1", 0.9) #adam_trainer g
hc.set("trainer.adam.generator.beta2", 0.999) #adam_trainer g
hc.set("trainer.rmsprop.discriminator.lr", 3e-5) # d learning rate
hc.set("trainer.rmsprop.generator.lr", 1e-4) # d learning rate
hc.set('trainer.slowdown.discriminator.d_fake_min', [0.12]) # healthy above this number on d_fake
hc.set('trainer.slowdown.discriminator.d_fake_max', [0.12001]) # unhealthy below this number on d_fake
hc.set('trainer.slowdown.discriminator.slowdown', [5]) # Divides speed by this number when unhealthy(d_fake low)
hc.set("trainer.sgd_adam.discriminator.lr", 3e-4) # d learning rate
hc.set("trainer.sgd_adam.generator.lr", 1e-3) # g learning rate

# Discriminator configuration
hc.set("discriminator", pyramid_nostride_discriminator.discriminator)
hc.set("discriminator.activation", [lrelu])#prelu("d_")])
hc.set('discriminator.regularizers.layer', layer_norm_1) # Size of fully connected layers

hc.set('discriminator.fc_layer', [False]) #If true, include a fully connected layer at the end of the discriminator
hc.set('discriminator.fc_layers', [0])# Number of fully connected layers to include
hc.set('discriminator.fc_layer.size', 378) # Size of fully connected layers

hc.set("discriminator.pyramid.layers", 5) #Layers in D
hc.set("discriminator.pyramid.depth_increase", 2)# Size increase of D's features on each layer

hc.set('discriminator.painters.layers', 2) #TODO has this ever worked?
hc.set('discriminator.painters.transitions', 5)
hc.set('discriminator.painters.activation', lrelu)

hc.set('discriminator.densenet.k', 32) #k is the number of features that are appended on each conv pass
hc.set('discriminator.densenet.layers', 1) #number of times to conv before size transition
hc.set('discriminator.densenet.transitions', 8) #number of transitions

hc.set('discriminator.add_noise', [True]) #add noise to input
hc.set('discriminator.noise_stddev', [1e-1]) #the amount of noise to add - always centered at 0
hc.set('discriminator.regularizers', [[minibatch_regularizer.get_features]]) # these regularizers get applied at the end of D

hc.set("sampler", progressive_enhancement_sampler.sample) # this is our sampling method.  Some other sampling ideas include cosine distance or adverarial encoding(not implemented but contributions welcome).
hc.set("sampler.samples", 3) # number of samples to generate at the end of each epoch
#hc.set('encoder', random_encoder.encode) # how to encode z
hc.set('encoder', random_combo_encoder.encode_gaussian) # how to encode z

#hc.set("g_mp3_dilations",[[1,2,4,8,16,32,64,128,256]])
#hc.set("g_mp3_filter",[3])
#hc.set("g_mp3_residual_channels", [8])
#hc.set("g_mp3_dilation_channels", [16])
#hc.set("mp3_seconds", args.seconds)
#hc.set("mp3_bitrate", args.bitrate)
#hc.set("mp3_size", args.seconds*args.bitrate)

#TODO vae
#hc.set("transfer_fct", [tf.nn.elu, tf.nn.relu, tf.nn.relu6, lrelu]);
#hc.set("e_activation", [tf.nn.elu, tf.nn.relu, tf.nn.relu6, lrelu]);
#hc.set("e_last_layer", [tf.nn.tanh]);
#hc.set('f_skip_fc', False)
#hc.set('f_hidden_1', 512)#list(np.arange(256, 512)))
#hc.set('f_hidden_2', 256)#list(np.arange(256, 512)))

#TODO audio
hc.set("model", "faces:1.0")

hc.set("examples_per_epoch", 30000/4)

#TODO category/bernouilli
categories = [[]]#[[2]+[2]+build_categories_config(30)]
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
hc.set("d_kernels", list(np.arange(10, 20)))
hc.set("d_kernel_dims", list(np.arange(100, 200)))

#TODO remove and replace with losses
hc.set("loss", ['custom'])

#TODO Vae Loss
hc.set("adv_loss", [False])
hc.set("latent_loss", [False])
hc.set("latent_lambda", list(np.linspace(.01, .1, num=30)))

#TODO Is this about adding z to the D?  Is this right? Investigate
hc.set("d_project", ['tiled'])

hc.set("batch_size", args.batch_size)

batch_no = 0

def frame_sample(sample_file, sess, config):
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

def epoch(sess, config):
    batch_size = config["batch_size"]
    n_samples =  config['examples_per_epoch']
    total_batch = int(n_samples / batch_size)
    global batch_no
    for i in range(total_batch):
        sample_file="samples/grid-%06d.png" % (batch_no * total_batch + i)
        frame_sample(sample_file, sess, config)

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
    if args.use_hc_io:
        hc.io.measure(config, measurements)
        hc.io.sample(config, sample_list)
    else:
        print("Offline sample created:", sample_list)


def get_function(name):
    if "lib." in name:
        name = name.replace("lib.", "hypergan.")
    if name == "function:hypergan.util.ops.prelu_internal":
        return prelu("g_")

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


def output_graph_size():
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

def run(args):
    crop = args.crop
    channels = int(args.size.split("x")[2])
    width = int(args.size.split("x")[0])
    height = int(args.size.split("x")[1])
    loadedFromSave = False

    print("[hypergan] Welcome back.  You are one of ", hc.count_configs(), " possible configurations.")
    for config in hc.configs(1):
        other_config = copy.copy(config)
        # load_saved_checkpoint(config)
        if(args.config):
            print("[hypergan] Creating or loading configuration in ~/.hypergan/configs/", args.config)

            config_path = os.path.expanduser('~/.hypergan/configs/'+args.config+'.json')
            print("Loading "+config_path)
            config = hc.load_or_create_config(config_path, config)

        config = lookup_functions(config)
        config['batch_size']=args.batch_size

        config['dtype']=other_config['dtype']#TODO: add this as a CLI argument, i.e "-e 'dtype=function:tf.float16'"
        config['trainer.rmsprop.discriminator.lr']=other_config['trainer.rmsprop.discriminator.lr']

        # Initialize tensorflow
        with tf.device(args.device):
            sess = tf.Session(config=tf.ConfigProto())

        with tf.device('/cpu:0'):
            #TODO don't branch on format
            if(args.format == 'mp3'):
                x,y, num_labels,examples_per_epoch = hypergan.loaders.audio_loader.mp3_tensors_from_directory(args.directory,config['batch_size'], seconds=args.seconds, channels=channels, bitrate=args.bitrate, format=args.format)
                f = None
            else:
                x,y, f, num_labels,examples_per_epoch = hypergan.loaders.image_loader.labelled_image_tensors_from_directory(args.directory,config['batch_size'], channels=channels, format=args.format,crop=crop,width=width,height=height)

        config['y_dims']=num_labels
        config['x_dims']=[height,width] #TODO can we remove this?
        config['channels']=channels

        if args.config is None:
            filename = '~/.hypergan/configs/'+config['uuid']+'.json'
            print("[hypergan] Saving network configuration to: " + filename)
            config = hc.load_or_create_config(filename, config)

        with tf.device(args.device):
            y=tf.one_hot(tf.cast(y,tf.int64), config['y_dims'], 1.0, 0.0)

            if(args.method == 'build' or args.method == 'serve'):
                graph = create_generator(config,x,y,f)
            else:
                graph = create(config,x,y,f)

        #TODO can we not do this?  might need to be after hc.io refactor
        if('parent_uuid' in config):
            save_file = "~/.hypergan/saves/"+config["parent_uuid"]+".ckpt"
            samples_path = "~/.hypergan/samples/"+config['parent_uuid']
        else:
            save_file = "~/.hypergan/saves/"+config["uuid"]+".ckpt"
            samples_path = "~/.hypergan/samples/"+config['uuid']
        if args.config:
            save_file = "~/.hypergan/saves/"+args.config+".ckpt"
            samples_path = "~/.hypergan/samples/"+args.config

        save_file = os.path.expanduser(save_file)
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        os.makedirs(os.path.expanduser(samples_path), exist_ok=True)
        build_file = os.path.expanduser("~/.hypergan/builds/"+config['uuid']+"/generator.ckpt")
        os.makedirs(os.path.dirname(build_file), exist_ok=True)


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

        output_graph_size()
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
                    if(not epoch(sess, config)):
                        print("Epoch failed")
                        break
                print("Checking save "+ str(i))
                if(args.save_every != 0 and i % args.save_every == args.save_every-1):
                    print(" |= Saving network")
                    saver = tf.train.Saver()
                    saver.save(sess, save_file)
                end_time = time.time()
                test_epoch(i, sess, config, start_time, end_time)

            tf.reset_default_graph()
            sess.close()

if __name__ == "__main__":
    run(args) 
    

