import argparse
import os
import uuid
import random
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
    parser.add_argument('--config_list', '-m', type=str, default=None, help='config list name')
    parser.add_argument('--use_hc_io', '-9', dest='use_hc_io', action='store_true', help='experimental')
    parser.add_argument('--add_full_image', type=bool, default=False, help='Instead of just the black and white X, add the whole thing.')
    return parser.parse_args()

args = parse_args()

width = int(args.size.split("x")[0])
height = int(args.size.split("x")[1])
channels = int(args.size.split("x")[2])

selector = hg.config.selector(args)

config = selector.random_config()
config_file = args.config

if args.config_list is not None:
    lines = tuple(open(args.config_list, 'r'))
    config_file = random.choice(lines).strip()
    print("config list chosen", config_file)

config_filename = os.path.expanduser('~/.hypergan/configs/'+config_file+'.json')
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

xa = x
xb = tf.tile(tf.image.rgb_to_grayscale(xa), [1,1,1,3])

def generator_config():
    return align_generator.config(
            z_projection_depth=[128],
            activation=[tf.nn.relu,tf.tanh,lrelu,resize_conv_generator.generator_prelu, tf.nn.crelu],
            final_activation=[None,tf.nn.tanh,resize_conv_generator.minmax],
            depth_reduction=[32],
            layer_filter=None,
            layer_regularizer=[None],
            block=[resize_conv_generator.standard_block, resize_conv_generator.inception_block, resize_conv_generator.dense_block, resize_conv_generator.repeating_block],
	    orthogonal_initializer_gain= list(np.linspace(0.1, 2, num=100))
    )

def discriminator_config():
    return hg.discriminators.align_discriminator.config(
	    activation=[tf.nn.relu, lrelu, tf.nn.relu6, tf.nn.elu, tf.nn.crelu, tf.tanh],
	    block_repeat_count=[1,2,3],
        block=[hg.discriminators.common.repeating_block,
               hg.discriminators.common.standard_block,
               hg.discriminators.common.strided_block
               ],
            depth_increase=[32],
            final_activation=[tf.nn.relu, tf.tanh, tf.nn.crelu, resize_conv_generator.minmax],
            layer_regularizer=[None],
            layers=[5,4,3],
	    extra_layers=[0,1,2,3],
	    extra_layers_reduction=[1,2,4],
            fc_layer_size=[300],
            fc_layers=[0,1],
            first_conv_size=[32],
            noise=[False, 1e-2],
            progressive_enhancement=[False, True],
			foundation= "additive",
	    orthogonal_initializer_gain= list(np.linspace(0.1, 2, num=100)),
        distance=[hg.discriminators.autoencoder_discriminator.l1_distance, hg.discriminators.autoencoder_discriminator.l2_distance],
        include_ae=[True, False],
        include_gs=[True, False],
        include_xabba=[True, False],
        include_gaab=[True, False],
        include_ga=[True, False],
        include_cross_distance=[True, False],
        include_cross_distance2=[True, False],
        include_encoded=[True, False]
    )

def generator_autoencode_config():
    return resize_conv_generator.config(
            z_projection_depth=[128],
            activation=[tf.nn.relu,tf.tanh,lrelu,resize_conv_generator.generator_prelu, tf.nn.crelu],
            final_activation=[None,tf.nn.tanh,resize_conv_generator.minmax],
            depth_reduction=[32],
            layer_filter=None,
            layer_regularizer=[None],
	    block_repeat_count=[1,2,3],
            block=[resize_conv_generator.standard_block, resize_conv_generator.inception_block, resize_conv_generator.dense_block, resize_conv_generator.repeating_block],
	    orthogonal_initializer_gain= list(np.linspace(0.1, 2, num=100))
    )

def loss_options():
    return {
        alignment_lambda: list(np.linspace(0.001, 10, num=100)),
        include_recdistance:[True, False],
        include_recdistance2:[True, False],
        include_grecdistance:[True, False],
        include_grecdistance2:[True, False],
        include_distance:[True, False]
    }

config['y_dims']=num_labels
config['x_dims']=[height,width]
config['channels']=channels
config['model']=args.config
config = hg.config.lookup_functions(config)
config['generator']=generator_config()
config['generator_autoencode']=generator_autoencode_config()
config['discriminators']=[discriminator_config()]
config['encoders']=[{"create": hg.encoders.match_discriminator_encoder.create}]

loss_selector = hg.config.selector(loss_options)
loss_config = loss_selector.random_config()
config['losses'][0].update(loss_config)

config_name="alignment-"+str(uuid.uuid4())
config_filename = os.path.expanduser('~/.hypergan/configs/'+config_name+'.json')
print("Saving config to ", config_filename)

selector.save(config_filename, config)
initial_graph = {
    'x':x,
    'xa':xa,
    'xb':xb,
    'y':y,
    'f':f,
    'num_labels':num_labels,
    'examples_per_epoch':examples_per_epoch
}

print("Config is ", config.trainer)

gan = hg.GAN(config, initial_graph)

gan.initialize_graph()

tf.train.start_queue_runners(sess=gan.sess)
exa, exb, static_x, static_z = gan.sess.run([gan.graph.xa, gan.graph.xb, gan.graph.x, gan.graph.z[0]])

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

accuracies = {
    "xa_to_rxa":accuracy(exa, gan.graph.rxa),
    "xb_to_rxb":accuracy(exb, gan.graph.rxb),
    "xa_to_rxabba":accuracy(exa, gan.graph.rxabba),
    "xb_to_rxbaab":accuracy(exb, gan.graph.rxbaab),
    "xa_to_xabba":accuracy(exa, gan.graph.xabba),
    "xb_to_xbaab":accuracy(exb, gan.graph.xbaab),

    # This only works because xb is grayscale of xa,
    "xb_to_xab":accuracy(exb, gan.graph.xab)

}

diversities={
    'rxa': batch_diversity(gan.graph.rxa),
    'rxb': batch_diversity(gan.graph.rxb),
    'rxabba': batch_diversity(gan.graph.rxabba),
    'rxbaab': batch_diversity(gan.graph.rxbaab),
    'rxab': batch_diversity(gan.graph.rxab),
    'rxba': batch_diversity(gan.graph.rxba),
    'xab': batch_diversity(gan.graph.xab),
    'xba': batch_diversity(gan.graph.xba),
    'xabba': batch_diversity(gan.graph.xabba),
    'xbaab': batch_diversity(gan.graph.xbaab)
}

diversities_items= list(diversities.items())
accuracies_items= list(accuracies.items())

sums = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
names = []

for i in range(40000):
    d_loss, g_loss = gan.train({gan.graph.xa: static_x, gan.graph.z[0]: static_z})
    if(np.abs(g_loss) > 10000):
        print("OVERFLOW");
        break

    if i % 100 == 0 and i != 0: 
        if 'k' in gan.graph:
            k = gan.sess.run([gan.graph.k], {gan.graph.xa: exa, gan.graph.z[0]: static_z})
            print("K", k, "d_loss", d_loss)
            if math.isclose(k[0], 0.0) or np.isnan(d_loss):
                sums = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                names = ["error k or dloss"]
                break
        if i > 500:
            diversities_v = gan.sess.run([v for _, v in diversities_items])
            accuracies_v = gan.sess.run([v for _, v in accuracies_items])
            print("D", diversities_v)
            broken = False
            for k, v in enumerate(diversities_v):
                sums[k] += v 
                name = diversities_items[k][0]
                names.append(name)
                if(np.abs(v) < 20000):
                    sums = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                    names = ["error diversity "+name]
                    broken = True
                    break

            for k, v in enumerate(accuracies_v):
                sums[k+len(diversities_items) ] += v 
                name = accuracies_items[k][0]
                names.append(name)
                if(np.abs(v) > 800):
                    sums = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                    names = ["error accuracy "+name]
                    broken = True
                    break
            print(sums)

            if(broken):
                break

        

with open("results-alignment", "a") as myfile:
    myfile.write(config_name+","+",".join(names)+"\n")
    myfile.write(config_name+","+",".join(["%.2f" % sum for sum in sums])+"\n")
 
tf.reset_default_graph()
gan.sess.close()
