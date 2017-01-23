import argparse
import os
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
from hypergan.loaders import *
from hypergan.samplers.common import *
from hypergan.util.globals import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train a colorizer!', add_help=True)
    parser.add_argument('directory', action='store', type=str, help='The location of your data.  Subdirectories are treated as different classes.  You must have at least 1 subdirectory.')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Number of samples to include in each batch.  If using batch norm, this needs to be preserved when in server mode')
    parser.add_argument('--crop', type=bool, default=False, help='If your images are perfectly sized you can skip cropping.')
    parser.add_argument('--device', '-d', type=str, default='/gpu:0', help='In the form "/gpu:0", "/cpu:0", etc.  Always use a GPU (or TPU) to train')
    parser.add_argument('--format', '-f', type=str, default='png', help='jpg or png')
    parser.add_argument('--sample_every', type=int, default=50, help='Samples the model every n epochs.')
    parser.add_argument('--save_every', type=int, default=30000, help='Saves the model every n epochs.')
    parser.add_argument('--size', '-s', type=str, default='64x64x3', help='Size of your data.  For images it is widthxheightxchannels.')
    parser.add_argument('--use_hc_io', type=bool, default=False, help='Set this to no unless you are feeling experimental.')
    return parser.parse_args()

def sampler(name, sess, config):
    generator = get_tensor("g")[0]
    y_t = get_tensor("y")
    z_t = get_tensor("z")
    x_t = get_tensor('x')
    fltr_x_t = get_tensor('xfiltered')
    x = sess.run([x_t])
    x = np.tile(x[0][0], [config['batch_size'],1,1,1])

    sample, bw_x = sess.run([generator, fltr_x_t], {x_t: x})
    bw = np.squeeze(np.tile(bw_x[0], [1,1,1,3]))
    stacks = []
    stacks.append([x[0], bw, sample[0], sample[1], sample[2], sample[3]])
    for i in range(4):
        stacks.append([sample[i*6+4+j] for j in range(6)])
    
    print('bwxshape', bw.shape, x[0].shape)
    images = np.vstack([np.hstack(s) for s in stacks])
    plot(config, images, name)

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

selector = hg.config.selector(args)

config = selector.random_config()
config_filename = os.path.expanduser('~/.hypergan/configs/colorizer.json')
config = selector.load_or_create_config(config_filename, config)

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
config['model']='colorizer'
config = hg.config.lookup_functions(config)

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
        sample_file = "samples/"+str(i)+".png"
        gan.sample_to_file(sample_file, sampler=sampler)
        if args.use_hc_io:
            hc.io.sample(config, [{"image":sample_file, "label": 'sample'}]) 

tf.reset_default_graph()
self.sess.close()
