import argparse
import os
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
from hypergan.loaders import *
from hypergan.samplers.common import *
from hypergan.generators import *
from hypergan.util.ops import *
from hypergan.discriminators import *

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

def optimize_d(g, d, config, initial_graph):
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

def create_random_discriminator():
    return pyramid_discriminator.config(
            activation=[tf.nn.relu, lrelu, tf.nn.relu6, tf.nn.elu],
            depth_increase=[1.5,1.7,2,2.1],
            final_activation=[tf.nn.relu, tf.tanh, None],
            layer_regularizer=[batch_norm_1, layer_norm_1, None],
            layers=[4,5,6],
            fc_layer_size=[2048,1024,512,256],
            fc_layers=[0,1,2],
            noise=[False, 1e-2],
            progressive_enhancement=[True, False],
            strided=[True, False]
        )

def run_gan(gan, steps):
    d_class_loss = 0
    gan.initialize_graph()

    coord = tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=gan.sess, coord=coord)
    for i in range(steps):
        gan.train()
        if i > steps // 2:
            d_class_loss += gan.sess.run(gan.graph.d_class_loss)

    coord.request_stop()
    coord.join(threads)
    tf.reset_default_graph()
    gan.sess.close()

    return d_class_loss


if(int(config['y_dims']) > 1):
    print("[discriminator] Class loss is on.  Semi-supervised learning mode activated.")
    config['losses'].append(hg.losses.supervised_loss.config())
else:
    raise "Need class loss.  Try adding some classes to your dataset."

g1 = config['generator']
#d1 = create_random_discriminator()
d1 = selector.load('best_d.json')
gan1 = optimize_d(g1, d1, config, initial_graph)
d_class_loss1 = run_gan(gan1, args.steps)

while True:
    d2 = create_random_discriminator()
    gan2 = optimize_d(g1, d2, config, initial_graph)
    d_class_loss2 = run_gan(gan2, args.steps)
    print("d_class_loss1 %02f d_class_loss2 %02f" % (d_class_loss1, d_class_loss2))
    if d_class_loss2 < d_class_loss1 or np.isnan(d_class_loss1):
        d1 = d2
        d_class_loss1 = d_class_loss2

    print("Best D: ", d1)
    selector.save("best_d.json", d1)

