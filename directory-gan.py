import hyperchamber as hc
from shared.ops import *
from shared.util import *
from shared.gan import *
from shared.gan_server import *
#import shared.jobs as jobs
import shared.hc_tf as hc_tf
import shared
import json
import uuid
import time

import shared.data_loader
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

import argparse

parser = argparse.ArgumentParser(description='Runs the GAN.')
parser.add_argument('--load_config', type=str)
parser.add_argument('--epochs', type=int, default=10)

parser.add_argument('--channels', type=int, default=3)
parser.add_argument('--directory', type=str)
parser.add_argument('--no_stop', type=bool)
parser.add_argument('--crop', type=bool, default=True)

parser.add_argument('--width', type=int, default=64)
parser.add_argument('--height', type=int, default=64)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--format', type=str, default='png')
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--server', type=bool, default=False)
parser.add_argument('--save_every', type=int, default=0)
parser.add_argument('--device', type=str, default='/gpu:0')

args = parser.parse_args()
start=1e-3
end=1e-3

num=100
hc.set('pretrained_model', ['preprocess'])

hc.set('f_skip_fc', True)
hc.set('f_hidden_1', list(np.arange(512, 1024)))
hc.set('f_hidden_2', list(np.arange(512, 1024)))

hc.set('d_optim_strategy', ['g_adam'])
hc.set("g_learning_rate", list(np.linspace(1e-4,1e-3,num=100)))
hc.set("d_learning_rate", list(np.linspace(1e-4,1e-4,num=100)))

hc.set("model", "40k_overfit:1.1")


hc.set("optimizer", ['rmsprop'])

hc.set('rmsprop_lr', list(np.linspace(1e-5, 2e-5)))
hc.set('simple_lr', list(np.linspace(0.01, 0.012, num=100)))
hc.set('simple_lr_g', list(np.linspace(2,3, num=10)))

hc.set('momentum_lr', list(np.linspace(0.005, 0.01, num=100)))
hc.set('momentum', list(np.linspace(0.8, 0.9999, num=1000)))
hc.set('momentum_lr_g', list(np.linspace(1, 3, num=100)))

hc.set("transfer_fct", [tf.nn.elu, tf.nn.relu, tf.nn.relu6, lrelu]);
hc.set("d_activation", [tf.nn.elu, tf.nn.relu, tf.nn.relu6, lrelu]);
hc.set("g_activation", [tf.nn.elu, tf.nn.relu, tf.nn.relu6, lrelu]);
hc.set("e_activation", [tf.nn.elu, tf.nn.relu, tf.nn.relu6, lrelu]);
hc.set("g_last_layer", [tf.nn.tanh]);
hc.set("e_last_layer", [tf.nn.tanh]);
hc.set('d_add_noise', [True])
hc.set('d_noise', [1e-2])

hc.set("g_last_layer_resnet_depth", [0])
hc.set("g_last_layer_resnet_size", [1])


hc.set('g_last_layer_stddev', list(np.linspace(0.15,1,num=40)))
hc.set('g_batch_norm_last_layer', [False])
hc.set('d_batch_norm_last_layer', [False, True])
hc.set('e_batch_norm_last_layer', [False, True])

hc.set('g_resnet_depth', [16,8])
hc.set('g_resnet_filter', [3])

hc.set('g_huge_stride', [8])#[])
hc.set('g_huge_filter', [9])

hc.set('g_atrous', [False])
hc.set('g_atrous_filter', [3])

hc.set('d_resnet_depth', [0])
hc.set('d_resnet_filter', [3])
conv_g_layers = build_deconv_config(layers=3, start=3, end=4)
if(args.test):
    conv_g_layers = [[10, 3, 3]]
print('conv_g_layers', conv_g_layers)

conv_d_layers = build_conv_config(4, 3, 4)
if(args.test):
    conv_d_layers = [[10, 3, 3]]
print('conv_d_layers', conv_d_layers)

hc.set("conv_size", [3])
hc.set("d_conv_size", [3])
hc.set("e_conv_size", [3])
hc.set("conv_g_layers", conv_g_layers)
hc.set("conv_d_layers", conv_d_layers)

hc.set('d_conv_expand_restraint', [2])
hc.set('e_conv_expand_restraint', [2])

hc.set('include_f_in_d', [False])

g_encode_layers = [[32, 64,128,256,512, 1024], 
        [64,128,256,512,1024, 2048]]
if(args.test):
    g_encode_layers = [[10, 3, 3]]
hc.set("g_encode_layers", g_encode_layers)
hc.set("z_dim", list(np.arange(64,256)))

hc.set('z_dim_random_uniform', 0)#list(np.arange(32,64)))

categories = [[2,3,5],build_categories_config(5), [2]+build_categories_config(10), [2]+build_categories_config(20), [2]+build_categories_config(40)]
print(categories)
hc.set('categories', [[]])#categories)
hc.set('categories_lambda', list(np.linspace(.001, .01, num=100)))
hc.set('category_loss', [False])

hc.set('g_class_loss', [False])
hc.set('g_class_lambda', list(np.linspace(0.01, .1, num=30)))
hc.set('d_fake_class_loss', [False])

hc.set("regularize", [False])
hc.set("regularize_lambda", list(np.linspace(0.001, .01, num=30)))

hc.set("g_batch_norm", [True])
hc.set("d_batch_norm", [True])
hc.set("e_batch_norm", [True])

hc.set("g_encoder", [True])

hc.set('d_linear_layer', [True])
hc.set('d_linear_layers', list(np.arange(256, 512)))

hc.set("g_target_prob", list(np.linspace(.65 /2., .85 /2., num=100)))
hc.set("d_label_smooth", list(np.linspace(0.15, 0.35, num=100)))

hc.set("d_kernels", list(np.arange(10, 20)))
hc.set("d_kernel_dims", list(np.arange(100, 300)))

hc.set("loss", ['custom'])

hc.set("adv_loss", [True])

hc.set("mse_loss", [False])
hc.set("mse_lambda",list(np.linspace(.01, .1, num=30)))

hc.set("latent_loss", [True])
hc.set("latent_lambda", list(np.linspace(.01, .1, num=30)))
hc.set("g_dropout", list(np.linspace(0.6, 0.99, num=30)))

hc.set("g_project", ['linear'])
hc.set("d_project", ['tiled'])
hc.set("e_project", ['tiled'])

hc.set("v_train", ['generator'])

hc.set("g_post_res_filter", [3])

hc.set("d_pre_res_filter", [7])
hc.set("d_pre_res_stride", [7])

hc.set("d_pool", [False])

hc.set("batch_size", args.batch)
hc.set('bounds_d_fake_min', [0.2])
hc.set('bounds_d_fake_max', [0.4])
hc.set('bounds_d_fake_slowdown', [200])

def sample_input(sess, config):
    x = get_tensor("x")
    y = get_tensor("y")
    encoded = get_tensor("encoded")
    sample, encoded, label = sess.run([x, encoded, y])
    return sample[0], encoded[0], label[0]


def split_sample(n, d_fake_sig, sample, x_dims, channels):
    samples = []

    for s, d in zip(sample, d_fake_sig):
        samples.append({'sample':s,'d':d})
    #samples = sorted(samples, key=lambda x: (1-x['d']))

    [print("sample ", s['d']) for s in samples[0:n]]
    return [np.reshape(s['sample'], [x_dims[0],x_dims[1], channels]) for s in samples[0:n]]
    #return [np.reshape(sample[0+i:1+i], [x_dims[0],x_dims[1], channels]) for i in range(n)]
def samples(sess, config):
    generator = get_tensor("g")
    y = get_tensor("y")
    x = get_tensor("x")
    categories = get_tensor('categories')
    d_fake_sigmoid = get_tensor("d_fake_sigmoid")
    rand = np.random.randint(0,config['y_dims'], size=config['batch_size'])
    #rand = np.zeros_like(rand)
    random_one_hot = np.eye(config['y_dims'])[rand]
    random_categories = [random_category(config['batch_size'], size) for size in config['categories']]
    #if(len(random_categories) > 0):
    #    random_categories = tf.concat(1, random_categories)
    #    random_categories = sess.run(random_categories)
    #    sample, d_fake_sig = sess.run([generator, d_fake_sigmoid], feed_dict={y:random_one_hot, categories: random_categories})
    #else:
    sample, d_fake_sig = sess.run([generator, d_fake_sigmoid], feed_dict={y:random_one_hot})
    #sample =  np.concatenate(sample, axis=0)
    return split_sample(10, d_fake_sig, sample, config['x_dims'], config['channels'])

def plot_mnist_digit(config, image, file):
    """ Plot a single MNIST image."""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    #plt.suptitle(config)
    plt.savefig(file)

def epoch(sess, config):
    batch_size = config["batch_size"]
    n_samples =  config['examples_per_epoch']
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        d_loss, g_loss = train(sess, config)
        if(i > 10 and not args.no_stop):
        
            if(math.isnan(d_loss) or math.isnan(g_loss) or g_loss > 1000 or d_loss > 1000):
                return False
        
            g = get_tensor('g')
            rX = sess.run([g])
            rX = np.array(rX)
            if(np.min(rX) < -1000 or np.max(rX) > 1000):
                return False

        #jobs.process(sess)

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

def test_epoch(epoch, j, sess, config, start_time, end_time):
    x, encoded, label = sample_input(sess, config)
    sample_file = "samples/input-"+str(j)+".png"
    plot(config, x, sample_file)
    encoded_sample = "samples/encoded-"+str(j)+".png"
    plot(config, encoded, encoded_sample)

    def to_int(one_hot):
        i = 0
        for l in list(one_hot):
            if(l>0.5):
                return i
            i+=1
        return None
    
    sample_file = {'image':sample_file, 'label':json.dumps(to_int(label))}
    encoded_sample = {'image':encoded_sample, 'label':'reconstructed'}
    
    sample = []
    sample += samples(sess, config)
    sample_list = [sample_file, encoded_sample]
    for s in sample:
        sample_file = "samples/config-"+str(j)+".png"
        plot(config, s, sample_file)
        sample_list.append({'image':sample_file,'label':'sample-'+str(j)})
        j+=1
    print("Creating sample")
    measurements = collect_measurements(epoch, sess, config, end_time - start_time)
    hc.io.measure(config, measurements)
    hc.io.sample(config, sample_list)
    return j

def record_run(config):
    results = test_config(sess, config)
    loss = np.array(results)
    #results = np.reshape(results, [results.shape[1], results.shape[0]])
    g_loss = [g for g,_,_,_ in loss]
    g_loss = np.mean(g_loss)
    d_fake = [d_ for _,d_,_,_ in loss]
    d_fake = np.mean(d_fake)
    d_real = [d for _,_,d,_ in loss]
    d_real = np.mean(d_real)
    e_loss = [e for _,_,_,e in loss]
    e_loss = np.mean(e_loss)

    # calculate D.difficulty = reduce_mean(d_loss_fake) - reduce_mean(d_loss_real)
    difficulty = d_real * (1-d_fake)
    # calculate G.ranking = reduce_mean(g_loss) * D.difficulty
    ranking = g_loss * (1.0-difficulty)

    ranking = e_loss
    results =  {
        'difficulty':float(difficulty),
        'ranking':float(ranking),
        'g_loss':float(g_loss),
        'd_fake':float(d_fake),
        'd_real':float(d_real),
        'e_loss':float(e_loss)
    }
    print("Recording ", results)
    #hc.io.record(config, results)





print("Generating configs with hyper search space of ", hc.count_configs())

j=0
k=0

def get_function(name):
    if not isinstance(name, str):
        return name
    print('name', name);
    if(name == "function:tensorflow.python.ops.gen_nn_ops.relu"):
        return tf.nn.relu
    if(name == "function:tensorflow.python.ops.nn_ops.relu"):
        return tf.nn.relu
    if(name == "function:tensorflow.python.ops.gen_nn_ops.relu6"):
        return tf.nn.relu6
    if(name == "function:tensorflow.python.ops.nn_ops.relu6"):
        return tf.nn.relu6
    if(name == "function:tensorflow.python.ops.gen_nn_ops.elu"):
        return tf.nn.elu
    if(name == "function:tensorflow.python.ops.nn_ops.elu"):
        return tf.nn.elu
    if(name == "function:tensorflow.python.ops.math_ops.tanh"):
        return tf.nn.tanh
    return eval(name.split(":")[1])
for config in hc.configs(1):
    other_config = copy.copy(config)
    if(args.load_config):
        print("Loading config", args.load_config)
        config.update(hc.io.load_config(args.load_config))
        if(not config):
            print("Could not find config", args.load_config)
            break
    config['g_activation']=get_function(config['g_activation'])
    config['d_activation']=get_function(config['d_activation'])
    config['e_activation']=get_function(config['e_activation'])
    config['transfer_fct']=get_function(config['transfer_fct'])
    #config['last_layer']=get_function(config['last_layer'])
    config['g_last_layer']=get_function(config['g_last_layer'])
    config['e_last_layer']=get_function(config['e_last_layer'])
    config['conv_d_layers']=[int(x) for x in config['conv_d_layers']]
    config['conv_g_layers']=[int(x) for x in config['conv_g_layers']]
    config['g_encode_layers']=[int(x) for x in config['g_encode_layers']]
    config['bounds_d_fake_min'] = other_config['bounds_d_fake_min']
    #config['categories'] = other_config['categories']
    #config['e_conv_size']=other_config['e_conv_size']
    #config['conv_size']=other_config['conv_size']
    #config['z_dim']=other_config['z_dim']
    #config['mse_loss']=True#other_config['mse_loss']
    #config['categories']=other_config['categories'][5:]+[2,3,5,7,9]
    #config['d_learning_rate']=config['g_learning_rate']*2
    #config['g_learning_rate']=_config['g_learning_rate']
    #config['categories_lambda']=other_config['categories_lambda']
    #config['conv_d_layers']=other_config['conv_d_layers']
    #config['adv_loss']=other_config['adv_loss']
    print("TODO: ADVLOSS ")
    with tf.device(args.device):
        sess = tf.Session(config=tf.ConfigProto())
    channels = args.channels
    crop = args.crop
    width = args.width
    height = args.height
    with tf.device('/cpu:0'):
        train_x,train_y, f, num_labels,examples_per_epoch = shared.data_loader.labelled_image_tensors_from_directory(args.directory,config['batch_size'], channels=channels, format=args.format,crop=crop,width=width,height=height)
    config['y_dims']=num_labels
    config['x_dims']=[height,width]
    config['channels']=channels

    if(args.load_config):
        pass
    #config['d_linear_layers']=other_config['d_linear_layers']
    #config['conv_g_layers'].append(channels)
    config['examples_per_epoch']=examples_per_epoch
    x = train_x
    y = train_y
    with tf.device(args.device):
        y=tf.one_hot(tf.cast(train_y,tf.int64), config['y_dims'], 1.0, 0.0)

        graph = create(config,x,y,f)
    saver = tf.train.Saver()
    if('parent_uuid' in config):
        save_file = "saves/"+config["parent_uuid"]+".ckpt"
    else:
        save_file = "saves/"+config["uuid"]+".ckpt"
    if(save_file and os.path.isfile(save_file)):
        print(" |= Loading network from "+ save_file)
        config['uuid']=config['parent_uuid']
        ckpt = tf.train.get_checkpoint_state('saves')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, save_file)
            print("Model loaded")
        else:
            print("No checkpoint file found")
    else:
        print("Starting new graph", config)
        def mul(s):
            x = 1
            for y in s:
                x*=y
            return x
        def get_size(v):
            shape = [int(x) for x in v.get_shape()]
            size = mul(shape)
            return [v.name, size/1024./1024.]
        [print(get_size(i)) for i in tf.all_variables()]
        with tf.device(args.device):
            init = tf.initialize_all_variables()
            sess.run(init)

    tf.train.start_queue_runners(sess=sess)

    #jobs.create_connection()
    if args.server:
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
            j=test_epoch(i, j, sess, config, start_time, end_time)
            if(i == args.epochs-1):
                print("Recording run...")
                record_run(config)

        tf.reset_default_graph()
        sess.close()


