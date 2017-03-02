import argparse
import os
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
from hypergan.loaders import *
from hypergan.samplers.common import *

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
    parser.add_argument('--config', '-c', type=str, default='colorizer', help='config name')
    parser.add_argument('--use_hc_io', '-9', dest='use_hc_io', action='store_true', help='experimental')
    parser.add_argument('--add_full_image', type=bool, default=False, help='Instead of just the black and white X, add the whole thing.')
    return parser.parse_args()

x_v = None
z_v = None
def sampler(gan, name):
    generator = gan.graph.g[0]
    y_t = gan.graph.y
    z_t = gan.graph.z[0]
    x_t = gan.graph.x
    sess = gan.sess
    config = gan.config
    global x_v
    global z_v
    x_v = None
    if(x_v == None):
        x_v, z_v = sess.run([x_t, z_t])
        x_v = np.tile(x_v[0], [config['batch_size'],1,1,1])
    #z_v = np.mgrid[-0.999:0.999:0.5, -0.999:0.999:0.25].reshape(2,-1).T

    sample, = sess.run([generator], {x_t: x_v, z_t: z_v})
    stacks = []
    stacks.append([x_v[1], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7]])
    for i in range(3):
        stacks.append([sample[i*8+8+j] for j in range(8)])
    
    images = np.vstack([np.hstack(s) for s in stacks])
    plot(config, images, name)

def add_bw(gan, net):
    x = gan.graph.x
    s = [int(x) for x in net.get_shape()]
    shape = [s[1], s[2]]
    x = tf.image.resize_images(x, shape, 1)
    
    if not gan.config.add_full_image:
        print( "[colorizer] Adding black and white image", x)
        x = tf.image.rgb_to_grayscale(x)
    else:
        print( "[colorizer] Adding full image", x)
        
    return x

def add_original_x(gan, net):
    x = gan.graph.x
    s = [int(x) for x in net.get_shape()]
    shape = [s[1], s[2]]
    x = tf.image.resize_images(x, shape, 1)
    print("Created bw ", x)

    x = tf.image.rgb_to_grayscale(x)
    #x = tf.nn.dropout(x, 0.005)
    #x += tf.random_normal(x.get_shape(), mean=0, stddev=1e-1, dtype=config['dtype'])

    return x

args = parse_args()

width = int(args.size.split("x")[0])
height = int(args.size.split("x")[1])
channels = int(args.size.split("x")[2])

selector = hg.config.selector(args)

config = selector.random_config()
config_filename = os.path.expanduser('~/.hypergan/configs/'+args.config+'.json')
config = selector.load_or_create_config(config_filename, config)

#TODO add this option to D
#TODO add this option to G
config['generator']['layer_filter'] = add_bw
config['discriminators'][0]['layer_filter'] = None#add_original_x

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

config['y_dims']=num_labels
config['x_dims']=[height,width]
config['channels']=channels
config['model']=args.config
config = hg.config.lookup_functions(config)

initial_graph = {
    'x':x,
    'y':y,
    'f':f,
    'num_labels':num_labels,
    'examples_per_epoch':examples_per_epoch
}

gan = hg.GAN(config, initial_graph)

save_file = os.path.expanduser("~/.hypergan/saves/"+args.config+".ckpt")
gan.load_or_initialize_graph(save_file)

tf.train.start_queue_runners(sess=gan.sess)
for i in range(10000000):
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
sess.close()
