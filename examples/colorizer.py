import argparse
import os
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import numpy as np
from hypergan.viewer import GlobalViewer
from hypergan.samplers.base_sampler import BaseSampler

from hypergan.gans.alpha_gan import AlphaGAN

x_v = None
z_v = None

class Sampler(BaseSampler):
    def sample(self, path):
        gan = self.gan
        generator = gan.generator.sample
        z_t = gan.uniform_encoder.sample
        x_t = gan.inputs.x
        
        sess = gan.session
        config = gan.config
        global x_v
        global z_v
        x_v, z_v = sess.run([x_t, z_t])
        x_v = np.tile(x_v[0], [gan.batch_size(),1,1,1])
        
        sample = sess.run(generator, {x_t: x_v, z_t: z_v})
        stacks = []
        bs = gan.batch_size()
        width = 5
        print(np.shape(x_v), np.shape(sample))
        stacks.append([x_v[1], sample[1], sample[2], sample[3], sample[4]])
        for i in range(bs//width-1):
            stacks.append([sample[i*width+width+j] for j in range(width)])
        images = np.vstack([np.hstack(s) for s in stacks])

        self.plot(images, path, False)
        return [{'images': images, 'label': 'tiled x sample'}]

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

def add_bw(gan, net):
    x = gan.inputs.x
    s = [int(x) for x in net.get_shape()]
    shape = [s[1], s[2]]
    x = tf.image.resize_images(x, shape, 1)
    
    if not gan.config.add_full_image:
        print( "[colorizer] Adding black and white image", x)
        x = tf.image.rgb_to_grayscale(x)
    else:
        print( "[colorizer] Adding full image", x)
        
    return x

args = parse_args()

width = int(args.size.split("x")[0])
height = int(args.size.split("x")[1])
channels = int(args.size.split("x")[2])

config = hg.configuration.Configuration.load(args.config+".json")

config.generator['layer_filter'] = add_bw

inputs = hg.inputs.image_loader.ImageLoader(args.batch_size)
inputs.create(args.directory,
              channels=channels, 
              format=args.format,
              crop=args.crop,
              width=width,
              height=height,
              resize=True)

gan = AlphaGAN(config, inputs=inputs)

gan.create()

tf.train.start_queue_runners(sess=gan.session)

GlobalViewer.enable()
config_name = args.config
title = "[hypergan] colorizer " + config_name
GlobalViewer.window.set_title(title)

sampler = Sampler(gan)

for i in range(10000000):
    gan.step()

    if i % args.save_every == 0 and i > 0:
        save_file = "save/model.ckpt"
        print("Saving " + save_file)
        gan.save(save_file)

    if i % args.sample_every == 0:
        print("Sampling "+str(i))
        sample_file = "samples/"+str(i)+".png"
        sampler.sample(sample_file)

tf.reset_default_graph()
gan.session.close()
