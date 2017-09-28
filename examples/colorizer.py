import os
import uuid
import random
import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import numpy as np
from hypergan.viewer import GlobalViewer
from hypergan.samplers.base_sampler import BaseSampler
from hypergan.samplers.random_walk_sampler import RandomWalkSampler
from hypergan.search.alphagan_random_search import AlphaGANRandomSearch
from common import *

from hypergan.gans.alpha_gan import AlphaGAN

x_v = None
z_v = None
layer_filter = None

class Sampler(BaseSampler):

    def sample(self, path, save_samples):
        gan = self.gan
        generator = gan.uniform_sample
        z_t = gan.uniform_encoder.sample
        x_t = gan.inputs.x
        
        sess = gan.session
        config = gan.config
        global x_v
        global z_v
        global layer_filter
        x_v = sess.run(x_t)
        x_v = np.tile(x_v[0], [gan.batch_size(),1,1,1])
        if layer_filter == None:
            layer_filter = gan.generator.layer_filter(x_t)
        
        layer_filter_v = sess.run(layer_filter, {x_t: x_v})

        sample = sess.run(generator, {x_t: x_v})
        stacks = []
        bs = gan.batch_size()
        width = 5
        print(np.shape(x_v[1]), np.shape(sample[2]), np.shape(layer_filter_v[1]))
        stacks.append([x_v[1], layer_filter_v[1], sample[2], sample[3], sample[4]])
 
        for i in range(bs//width-1):
            stacks.append([sample[i*width+width+j] for j in range(width)])
        images = np.vstack([np.hstack(s) for s in stacks])

        self.plot(images, path, save_samples)
        return [{'images': images, 'label': 'tiled x sample'}]

def apply_mask(gan, config, net,x=None):
    if x == None:
        x = gan.inputs.x
    filtered = net
    shape = gan.ops.shape(x)
    mask = tf.ones([shape[1], shape[2], shape[3]])
    mask = tf.greater(mask, 0)
    scaling = 0.6
    mask = tf.image.central_crop(mask, scaling)
    left = (shape[1]*scaling)//2 * 0.75
    top = (shape[2]*scaling)//2 * 0.75
    mask = tf.image.pad_to_bounding_box(mask, int(top), int(left), shape[1], shape[2])
    mask = tf.cast(mask, tf.float32)
    backmask = (1.0-mask) 
    filtered = backmask* x + mask * filtered
    print("FRAMING IMAGE", filtered) 
    return filtered

def add_bw(gan, config, net):
    x = gan.inputs.x
    s = [int(x) for x in net.get_shape()]
    print("S IS ", s)
    shape = [s[1], s[2]]
    x = tf.image.resize_images(x, shape, 1)
    bwnet = tf.slice(net, [0, 0, 0, 0], [s[0],s[1],s[2], 3])
    
    if not gan.config.add_full_image:
        print( "[colorizer] Adding black and white image", x)
        filtered = tf.image.rgb_to_grayscale(x)
        if config.blank_black_and_white is not None:
            filtered = tf.zeros_like(filtered)
        if config.colorizer_noise is not None:
            filtered += tf.random_normal(filtered.get_shape(), mean=0, stddev=config.colorizer_noise, dtype=tf.float32)

        if gan.config.add_full_image_frame:
            bw = tf.image.rgb_to_grayscale(filtered)
            bw = tf.tile(bw,[1,1,1,3])
            filtered = apply_mask(gan,config,bw, x)
    else:
        print( "[colorizer] Adding full image", x)
        filtered = x

    return filtered

arg_parser = ArgumentParser("Colorize an image")
arg_parser.add_image_arguments()
arg_parser.parser.add_argument('--add_full_image', type=bool, default=False, help='Instead of just the black and white X, add the whole thing.')
arg_parser.parser.add_argument('--add_full_image_frame', type=bool, default=False, help='Frame the corners of the image.  Incompatible with add_full_image.')
args = arg_parser.parse_args()

width, height, channels = parse_size(args.size)

config = lookup_config(args)
if args.action == 'search':
    random_config = AlphaGANRandomSearch({}).random_config()
    if args.config_list is not None:
        config = random_config_from_list(args.config_list)

        config["generator"]=random_config["generator"]
        config["g_encoder"]=random_config["g_encoder"]
        config["discriminator"]=random_config["discriminator"]
        config["z_discriminator"]=random_config["z_discriminator"]

        # TODO Other search terms?
    else:
        config = random_config
    config["cycloss_lambda"]= 0
    config["discriminator"]["skip_layer_filters"]=random.choice([[],[0],[0,1],[0,1,2],[0,1,2,3]])
    config["class"]="class:hypergan.gans.alpha_gan.AlphaGAN" # TODO

    config.generator['layer_filter'] = random.choice([None, add_bw])
    config.generator['final_layer_filter'] = random.choice([None, apply_mask])
    config.discriminator['layer_filter'] = random.choice([None, add_bw])
    config.g_encoder['layer_filter'] = random.choice([None, add_bw])

if args.add_full_image:
    config["add_full_image"]=True
if args.add_full_image_frame:
    config["add_full_image_frame"]=True

inputs = hg.inputs.image_loader.ImageLoader(args.batch_size)
inputs.create(args.directory,
        channels=channels, 
        format=args.format,
        crop=args.crop,
        width=width,
        height=height,
        resize=True)

save_file = "save/model.ckpt"

def setup_gan(config, inputs, args):
    gan = hg.GAN(config, inputs=inputs)

    if(args.action != 'search' and os.path.isfile(save_file+".meta")):
        gan.load(save_file)

    tf.train.start_queue_runners(sess=gan.session)

    config_name = args.config
    GlobalViewer.title = "[hypergan] colorizer " + config_name
    GlobalViewer.enabled = args.viewer

    return gan

def train(config, inputs, args):
    gan = setup_gan(config, inputs, args)
    sampler = lookup_sampler(args.sampler or Sampler)(gan)
    samples = 0

    metrics = [batch_accuracy(gan.inputs.x, gan.uniform_sample), batch_diversity(gan.uniform_sample)]
    sum_metrics = [0 for metric in metrics]
    for i in range(args.steps):
        gan.step()

        if args.action == 'train' and i % args.save_every == 0 and i > 0:
            print("saving " + save_file)
            gan.save(save_file)

        if i % args.sample_every == 0:
            sample_file="samples/%06d.png" % (samples)
            samples += 1
            sampler.sample(sample_file, args.save_samples)

        if i > args.steps * 9.0/10:
            for k, metric in enumerate(gan.session.run(metrics)):
                print("Metric "+str(k)+" "+str(metric))
                sum_metrics[k] += metric 

    tf.reset_default_graph()
    return sum_metrics

def sample(config, inputs, args):
    gan = setup_gan(config, inputs, args)
    sampler = lookup_sampler(args.sampler or RandomWalkSampler)(gan)
    samples = 0
    for i in range(args.steps):
        sample_file="samples/%06d.png" % (samples)
        samples += 1
        sampler.sample(sample_file, args.save_samples)

def search(config, inputs, args):
    metrics = train(config, inputs, args)

    config_filename = "colorizer-"+str(uuid.uuid4())+'.json'
    hc.Selector().save(config_filename, config)
    with open(args.search_output, "a") as myfile:
        myfile.write(config_filename+","+",".join([str(x) for x in metrics])+"\n")

if args.action == 'train':
    metrics = train(config, inputs, args)
    print("Resulting metrics:", metrics)
elif args.action == 'sample':
    sample(config, inputs, args)
elif args.action == 'search':
    search(config, inputs, args)
else:
    print("Unknown action: "+args.action)
