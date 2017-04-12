import argparse
import os
import uuid
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
from hypergan.loaders import *
from hypergan.util.hc_tf import *
from hypergan.samplers.common import *
from hypergan.generators import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train a colorizer!', add_help=True)
    parser.add_argument('directory', action='store', type=str, help='The location of your data.  Subdirectories are treated as different classes.  You must have at least 1 subdirectory.')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Number of samples to include in each batch.  If using batch norm, this needs to be preserved when in server mode')
    parser.add_argument('--crop', type=bool, default=False, help='If your images are perfectly sized you can skip cropping.')
    parser.add_argument('--device', '-d', type=str, default='/gpu:0', help='In the form "/gpu:0", "/cpu:0", etc.  Always use a GPU (or TPU) to train')
    parser.add_argument('--format', '-f', type=str, default='png', help='jpg or png')
    parser.add_argument('--sample_every', type=int, default=50, help='Samples the model every n epochs.')
    parser.add_argument('--size', '-s', type=str, default='64x64x3', help='Size of your data.  For images it is widthxheightxchannels.')
    parser.add_argument('--config', '-c', type=str, default='stable', help='config name')
    parser.add_argument('--use_hc_io', '-9', dest='use_hc_io', action='store_true', help='experimental')
    parser.add_argument('--add_full_image', type=bool, default=False, help='Instead of just the black and white X, add the whole thing.')
    return parser.parse_args()

args = parse_args()

width = int(args.size.split("x")[0])
height = int(args.size.split("x")[1])
channels = int(args.size.split("x")[2])

selector = hg.config.selector(args)

config = selector.random_config()
config_filename = os.path.expanduser('~/.hypergan/configs/'+args.config+'.json')
config = selector.load_or_create_config(config_filename, config)

# TODO refactor, shared in CLI
config['dtype']=tf.float32
config['batch_size'] = args.batch_size

if args.add_full_image:
    config['add_full_image']=args.add_full_image
    
x,y,f,num_labels,examples_per_epoch = image_loader.labelled_image_tensors_from_directory(
                        args.directory,
                        config['batch_size'], 
                        channels=channels, 
                        format=args.format,
                        crop=args.crop,
                        width=width,
                        height=height)

def generator_config():
    return resize_conv_generator.config(
            z_projection_depth=[32,64,128,256,512],
            activation=[tf.nn.relu,tf.tanh,lrelu,resize_conv_generator.generator_prelu, tf.nn.crelu],
            final_activation=[None,tf.nn.tanh,resize_conv_generator.minmax],
            depth_reduction=[64,32,24,16,128],
            layer_filter=None,
            layer_regularizer=[layer_norm_1,batch_norm_1,None],
	    block_repeat_count=[1,2,3],
	    batch_norm_momentum=[0.1,0.01,0.001,0.0001,1e-5,0],
	    batch_norm_epsilon=[1, 0.1, 0.01, 0.001, 1e-5, 0.5],
            block=[resize_conv_generator.standard_block, resize_conv_generator.inception_block, resize_conv_generator.dense_block, resize_conv_generator.repeating_block],
	    orthogonal_initializer_gain= list(np.linspace(0.1, 2, num=50)),
            resize_image_type=[1]
    )

def discriminator_config():
    return hg.discriminators.autoencoder_discriminator.config(
	    activation=[tf.nn.relu, lrelu, tf.nn.relu6, tf.nn.elu],
            depth_increase=[64,32,16,128],
            final_activation=[tf.nn.relu, tf.tanh, tf.nn.crelu],
            layer_regularizer=[batch_norm_1, layer_norm_1, None],
	    batch_norm_momentum=[0.1,0.01,0.001,0.0001,1e-5,0],
	    batch_norm_epsilon=[1, 0.1, 0.01, 0.001, 1e-5, 0.5],
            layers=[5,4,3],
	    extra_layers=[0,2,4],
	    extra_layers_reduction=[1,2,4],
            fc_layer_size=[150],
            fc_layers=[0],
            first_conv_size=[8,12,16],
            noise=[False, 1e-2],
            progressive_enhancement=[False, True],
			foundation= "additive",
	    orthogonal_initializer_gain= list(np.linspace(0.1, 2, num=50)),
        distance=[hg.discriminators.autoencoder_discriminator.l1_distance, hg.discriminators.autoencoder_discriminator.l2_distance],
            strided=[True]
    )



config['y_dims']=num_labels
config['x_dims']=[height,width]
config['channels']=channels
config['model']=args.config
config = hg.config.lookup_functions(config)
config['generator']=generator_config()
config['discriminators']=[discriminator_config()]


config_name="static-batch-"+str(uuid.uuid4())
config_filename = os.path.expanduser('~/.hypergan/configs/'+config_name+'.json')
print("Saving config to ", config_filename)

selector.save(config_filename, config)
initial_graph = {
    'x':x,
    'y':y,
    'f':f,
    'num_labels':num_labels,
    'examples_per_epoch':examples_per_epoch
}

gan = hg.GAN(config, initial_graph)

gan.initialize_graph()

tf.train.start_queue_runners(sess=gan.sess)
static_x, static_z = gan.sess.run([gan.graph.x, gan.graph.z[0]])

def batch_diversity(net):
    bs = int(net.get_shape()[0])
    avg = tf.reduce_mean(net, axis=0)

    s = [int(x) for x in avg.get_shape()]
    avg = tf.reshape(avg, [1, s[0], s[1], s[2]])

    tile = [1 for x in net.get_shape()]
    tile[0] = bs
    avg = tf.tile(avg, tile)
    net -= avg
    return tf.reduce_sum(tf.abs(net))

def accuracy(a, b):
    "Each point of a is measured against the closest point on b.  Distance differences are added together."
    difference = tf.abs(a-b)
    difference = tf.reduce_min(difference, axis=1)
    difference = tf.reduce_sum(difference, axis=1)
    return tf.reduce_sum( tf.reduce_sum(difference, axis=0) , axis=0) 


accuracy_x_to_g=accuracy(static_x, gan.graph.g[0])
diversity_g = batch_diversity(gan.graph.g[0])
diversity_x = batch_diversity(gan.graph.x)
diversity_diff = tf.abs(diversity_x - diversity_g)
ax_sum = 0
dd_sum = 0
dx_sum = 0

for i in range(6000):
    d_loss, g_loss = gan.train({gan.graph.x: static_x, gan.graph.z[0]: static_z})
    if(np.abs(g_loss) > 10000):
        print("OVERFLOW");
        ax_sum=10000000.00
        dd_sum=10000000.00
        dx_sum=10000000.00
        break

    if i % 100 == 0 and i != 0 and i > 400: 
        ax, dg, dx, dd = gan.sess.run([accuracy_x_to_g, diversity_g, diversity_x, diversity_diff], {gan.graph.x: static_x, gan.graph.z[0]: static_z})
        print("ERROR", ax, dg, dx, dd)
        if np.abs(ax) > 800.0 or np.abs(dg) < 20000 or np.isnan(d_loss):
            ax_sum =100000.00
            break

    if(i > 5900):
        ax, dg, dx, dd = gan.sess.run([accuracy_x_to_g, diversity_g, diversity_x, diversity_diff], {gan.graph.x: static_x, gan.graph.z[0]: static_z})
        ax_sum += ax
        dd_sum += dd
        

with open("results-static-batch-lsgan.csv", "a") as myfile:
    myfile.write(config_name+","+str(ax_sum)+","+str(dx_sum)+","+str(dd_sum)+"\n")
 
tf.reset_default_graph()
gan.sess.close()
