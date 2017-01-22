import argparse
import os
import tensorflow as tf
import hypergan as hg
from hypergan.loaders import *
from hypergan.samplers.common import *
from hypergan.util.globals import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train a colorizer!', add_help=True)
    parser.add_argument('directory', action='store', type=str, help='The location of your data.  Subdirectories are treated as different classes.  You must have at least 1 subdirectory.')
    parser.add_argument('--save_every', type=int, default=1000, help='Saves the model every n epochs.')
    parser.add_argument('--sample_every', type=int, default=50, help='Samples the model every n epochs.')
    parser.add_argument('--size', '-s', type=str, default='64x64x3', help='Size of your data.  For images it is widthxheightxchannels.')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Number of samples to include in each batch.  If using batch norm, this needs to be preserved when in server mode')
    parser.add_argument('--crop', type=bool, default=False, help='If your images are perfectly sized you can skip cropping.')
    parser.add_argument('--format', '-f', type=str, default='png', help='jpg or png')
    parser.add_argument('--device', '-d', type=str, default='/gpu:0', help='In the form "/gpu:0", "/cpu:0", etc.  Always use a GPU (or TPU) to train')
    return parser.parse_args()

def sampler(name, sess, config):
    generator = get_tensor("g")[0]
    y_t = get_tensor("y")
    z_t = get_tensor("z")
    x_t = get_tensor('x')
    fltr_x_t = get_tensor('xfiltered')

    sample, x, bw_x = sess.run([generator, x_t, fltr_x_t])#, categories_t: categories})
    bw = np.squeeze(np.tile(bw_x[0], [1,1,1,3]))
    stacks = [x[0], bw, sample[0]]
    print('bwxshape', bw.shape, x[0].shape)
    plot(config, np.hstack(stacks), name)

def add_bw(gan, net):
    x = get_tensor('x')
    s = [int(x) for x in net.get_shape()]
    shape = [s[1], s[2]]
    x = tf.image.resize_images(x, shape, 1)
    print("Created bw ", x)

    bw_net = tf.image.rgb_to_grayscale(x)
    return bw_net

args = parse_args()

width = int(args.size.split("x")[0])
height = int(args.size.split("x")[1])
channels = int(args.size.split("x")[2])

config = hg.config.random(args)

#TODO add this option to D
#TODO add this option to G
config['generator.layer_filter'] = add_bw

# TODO refactor, shared in CLI
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

config['y_dims']=num_labels
config['x_dims']=[height,width]
config['channels']=channels

initial_graph = {
    'x':x,
    'y':y,
    'f':f,
    'num_labels':num_labels,
    'examples_per_epoch':examples_per_epoch
}

gan = hg.GAN(config, initial_graph)

save_file = os.path.expanduser("~/.hypergan/saves/colorizer.ckpt")
gan.load_or_initialize_graph(save_file)

tf.train.start_queue_runners(sess=gan.sess)
for i in range(100000):
    d_loss, g_loss = gan.train()

    if i % args.save_every == 0 and i > 0:
        print("Saving " + save_file)
        gan.save(save_file)

    if i % args.sample_every == 0 and i > 0:
        print("Sampling "+str(i))
        gan.sample_to_file("samples/"+str(i)+".png", sampler=sampler)

tf.reset_default_graph()
self.sess.close()
