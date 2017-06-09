import argparse
import os
import uuid
import random
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc

from hypergan.search.random_search import RandomSearch

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
    return parser.parse_args()

args = parse_args()

width = int(args.size.split("x")[0])
height = int(args.size.split("x")[1])
channels = int(args.size.split("x")[2])


config_file = args.config

if args.config_list is not None:
    lines = tuple(open(args.config_list, 'r'))
    config_file = random.choice(lines).strip()
    print("config list chosen", config_file)

config_filename = os.path.expanduser('~/.hypergan/configs/'+config_file+'.json')
config = hc.Selector().load(config_filename)

inputs = hg.inputs.image_loader.ImageLoader(args.batch_size)
inputs.create(args.directory,
              channels=channels, 
              format=args.format,
              crop=args.crop,
              width=width,
              height=height,
              resize=False)

random_search = RandomSearch({})
config["generator"]=random_search.generator_config()
config["discriminator"]=random_search.discriminator_config()

config_name="static-batch-"+str(uuid.uuid4())
config_filename = os.path.expanduser('~/.hypergan/configs/'+config_name+'.json')
print("Saving config to ", config_filename)

hc.Selector().save(config_filename, config)

print("Creating GAN FROM ", config)

gan = hg.GAN(config, inputs=inputs)

gan.create()

tf.train.start_queue_runners(sess=gan.session)
static_x, static_z = gan.session.run([inputs.x, gan.encoder.sample])

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


accuracy_x_to_g=accuracy(static_x, gan.generator.sample)
diversity_g = batch_diversity(gan.generator.sample)
diversity_x = batch_diversity(gan.inputs.x)
diversity_diff = tf.abs(diversity_x - diversity_g)
ax_sum = 0
dd_sum = 0
dx_sum = 0
dg_sum = 0

for i in range(12000):
    gan.step({gan.inputs.x: static_x, gan.encoder.sample: static_z})

    if i % 100 == 0 and i != 0 and i > 400: 
        #if 'k' in gan.graph:
        #    k, ax, dg, dx, dd = gan.sess.run([gan.graph.k, accuracy_x_to_g, diversity_g, diversity_x, diversity_diff], {gan.graph.x: static_x, gan.graph.z[0]: static_z})
        #    print("ERROR", ax, dg, dx, dd, k)
        #    if math.isclose(k, 0.0) or np.abs(ax) > 800.0 or np.abs(dg) < 20000 or np.isnan(d_loss):
        #        ax_sum =100000.00
        #        break
        ax, dg, dx, dd = gan.session.run([accuracy_x_to_g, diversity_g, diversity_x, diversity_diff], {gan.inputs.x: static_x, gan.encoder.sample: static_z})
        print("ERROR", ax, dg, dx, dd)
        if np.abs(ax) > 800.0 or np.abs(dg) < 20000:
            ax_sum =100000.00
            break


    if(i > 11400):
        ax, dg, dx, dd = gan.session.run([accuracy_x_to_g, diversity_g, diversity_x, diversity_diff], {gan.inputs.x: static_x, gan.encoder.sample: static_z})
        ax_sum += ax
        dg_sum += dg
        dx_sum += dx
        dd_sum += dd
        

with open("results-static-batch", "a") as myfile:
    myfile.write(config_name+","+str(ax_sum)+","+str(dx_sum)+","+str(dg_sum)+","+str(dd_sum)+","+str(dx_sum + dg_sum)+"\n")
 
tf.reset_default_graph()
gan.session.close()
