import os
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import numpy as np
from hypergan.viewer import GlobalViewer
from hypergan.samplers.base_sampler import BaseSampler
from hypergan.samplers.random_walk_sampler import RandomWalkSampler
from common import *

from hypergan.gans.alpha_gan import AlphaGAN

x_v = None
z_v = None

class Sampler(BaseSampler):
    def sample(self, path, save_samples):
        gan = self.gan
        generator = gan.generator.sample
        z_t = gan.uniform_encoder.sample
        x_t = gan.inputs.x
        
        sess = gan.session
        config = gan.config
        global x_v
        global z_v
        x_v = sess.run(x_t)
        x_v = np.tile(x_v[0], [gan.batch_size(),1,1,1])
        
        sample = sess.run(generator, {x_t: x_v})
        stacks = []
        bs = gan.batch_size()
        width = 5
        print(np.shape(x_v), np.shape(sample))
        stacks.append([x_v[1], sample[1], sample[2], sample[3], sample[4]])
        for i in range(bs//width-1):
            stacks.append([sample[i*width+width+j] for j in range(width)])
        images = np.vstack([np.hstack(s) for s in stacks])

        self.plot(images, path, True)
        return [{'images': images, 'label': 'tiled x sample'}]

def add_bw(gan, net):
    x = gan.inputs.x
    s = [int(x) for x in net.get_shape()]
    print("S IS ", s)
    shape = [s[1], s[2]]
    x = tf.image.resize_images(x, shape, 1)
    bwnet = tf.slice(net, [0, 0, 0, 0], [s[0],s[1],s[2], 3])
    
    if not gan.config.add_full_image:
        print( "[colorizer] Adding black and white image", x)
        x = tf.image.rgb_to_grayscale(x)
        bwnet = tf.image.rgb_to_grayscale(bwnet)
        x = tf.concat(axis=3, values=[x, bwnet])
    else:
        print( "[colorizer] Adding full image", x)
        
    return x

arg_parser = ArgumentParser("Colorize an image")
arg_parser.add_image_arguments()
arg_parser.parser.add_argument('--add_full_image', type=bool, default=False, help='Instead of just the black and white X, add the whole thing.')
args = arg_parser.parse_args()


width = int(args.size.split("x")[0])
height = int(args.size.split("x")[1])
channels = int(args.size.split("x")[2])

config = hg.configuration.Configuration.load(args.config+".json")

config.generator['layer_filter'] = add_bw
config.discriminator['layer_filter'] = add_bw

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

save_file = "save/model.ckpt"
if(os.path.isfile(save_file+".meta")):
    sampler = RandomWalkSampler(gan)
    gan.load(save_file)
else:
    sampler = Sampler(gan)

for i in range(10000000):
    if args.action == 'train':
        gan.step()

        if i % args.save_every == 0 and i > 0:
            print("Saving " + save_file)
            gan.save(save_file)

    if args.action == "sample" or i % args.sample_every == 0:
        print("Sampling "+str(i))
        sample_file = "samples/"+str(i)+".png"
        sampler.sample(sample_file, False)

tf.reset_default_graph()

