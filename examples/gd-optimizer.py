import argparse
import os
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
from hypergan.loaders import *
from hypergan.samplers.common import *
from hypergan.generators import *
from hypergan.util.ops import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train a colorizer!', add_help=True)
    parser.add_argument('directory', action='store', type=str, help='The location of your data.  Subdirectories are treated as different classes.  You must have at least 1 subdirectory.')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Number of samples to include in each batch.  If using batch norm, this needs to be preserved when in server mode')
    parser.add_argument('--steps', '-z', type=int, default=10000000, help='number of steps to run for.  defaults to a lot')
    parser.add_argument('--crop', type=bool, default=False, help='If your images are perfectly sized you can skip cropping.')
    parser.add_argument('--device', '-d', type=str, default='/gpu:0', help='In the form "/gpu:0", "/cpu:0", etc.  Always use a GPU (or TPU) to train')
    parser.add_argument('--format', '-f', type=str, default='png', help='jpg or png')
    parser.add_argument('--sample_every', type=int, default=50, help='Samples the model every n epochs.')
    parser.add_argument('--save_every', type=int, default=30000, help='Saves the model every n epochs.')
    parser.add_argument('--size', '-s', type=str, default='64x64x3', help='Size of your data.  For images it is widthxheightxchannels.')
    parser.add_argument('--config', '-c', type=str, default='colorizer', help='config name')
    parser.add_argument('--use_hc_io', '-9', dest='use_hc_io', action='store_true', help='experimental')
    return parser.parse_args()

args = parse_args()

width = int(args.size.split("x")[0])
height = int(args.size.split("x")[1])
channels = int(args.size.split("x")[2])

selector = hg.config.selector(args)

config = selector.random_config()
config_filename = os.path.expanduser('~/.hypergan/configs/'+args.config+'.json')
config = selector.load_or_create_config(config_filename, config)

config['dtype']=tf.float32
config['batch_size'] = args.batch_size

x,y,f,num_labels,examples_per_epoch = image_loader.labelled_image_tensors_from_directory(
                            args.directory,
                            config['batch_size'], 
                            channels=channels, 
                            format=args.format,
                            crop=args.crop,
                            width=width,
                            height=height)
initial_graph = {
    'x':x,
    'y':y,
    'f':f,
    'num_labels':num_labels,
    'examples_per_epoch':examples_per_epoch
}


config['y_dims']=num_labels
config['x_dims']=[height,width]
config['channels']=channels
config['model']=args.config
print(config)
config = hg.config.lookup_functions(config)

def optimize_g(g, d, config, initial_graph):
    config['generator']=g
    config['discriminators']=[d]

    x,y,f,num_labels,examples_per_epoch = image_loader.labelled_image_tensors_from_directory(
                                args.directory,
                                config['batch_size'], 
                                channels=channels, 
                                format=args.format,
                                crop=args.crop,
                                width=width,
                                height=height)
    initial_graph = {
        'x':x,
        'y':y,
        'f':f,
        'num_labels':num_labels,
        'examples_per_epoch':examples_per_epoch
    }


    gan = hg.GAN(config, initial_graph)
    return gan

def create_random_generator():
    return resize_conv_generator.config(
            z_projection_depth=[1024,512,256,128],
            activation=[tf.nn.relu,tf.tanh,lrelu,resize_conv_generator.generator_prelu],
            final_activation=[None,tf.nn.tanh],
            depth_reduction=[2,1.5,2.1],
            layer_filter=None,
            layer_regularizer=[layer_norm_1,batch_norm_1]
    )

def run_gan(gan, steps):
    gan.initialize_graph()

    coord = tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=gan.sess, coord=coord)
    for i in range(steps):
        gan.train()

    d_log = gan.sess.run(gan.graph.d_log)

    coord.request_stop()
    coord.join(threads)
    tf.reset_default_graph()
    gan.sess.close()

    return d_log



g1 = create_random_generator()
d1 = config['discriminators'][0]
gan1 = optimize_g(g1, d1, config, initial_graph)
#optimize_d()
d_log1 = run_gan(gan1, args.steps)

while True:
    g2 = create_random_generator()
    gan2 = optimize_g(g2, d1, config, initial_graph)
    d_log2 = run_gan(gan2, args.steps)
    if d_log2 < d_log1 or np.isnan(d_log1):
        g1 = g2
        d_log1 = d_log2

    print("d_log1 %02f d_log2 %02f" % (d_log1, d_log2))
    print("Best G: ", g1)
    selector.save("best_gd.json", g1)

