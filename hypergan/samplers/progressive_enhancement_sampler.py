import tensorflow as tf
from hypergan.util.ops import *
from hypergan.samplers.common import *
import os
import json

# This sampler builds different images for each
# level of our GAN
# For instance, it will generate 256x256, 128x128, and 64x64 images for each z
def build_samples(gan):
    config = gan.config
    generator = gan.graph.g[0]
    gs = gan.graph.gs
    y = gan.graph.y
    x = gan.graph.x
    xs = gan.graph.xs
    categories = gan.graph.categories
    samples = 3
    rand = np.random.randint(0,config['y_dims'], size=config['batch_size'])
    #rand = np.zeros_like(rand)
    random_one_hot = np.eye(config['y_dims'])[rand]
    sample, sample2, sample3 = gan.sess.run([generator, gs[1], gs[2]], feed_dict={y:random_one_hot})
    a = split_sample(samples, sample, config['x_dims'], config['channels'])
    b = split_sample(samples, sample2, [gs[1].get_shape()[1], gs[1].get_shape()[2]], config['channels'])
    c = split_sample(samples, sample3, [gs[2].get_shape()[1], gs[2].get_shape()[2]], config['channels'])
    return [val for pair in zip(a, b, c) for val in pair]

def split_sample(n, sample, x_dims, channels):
    return [np.reshape(s, [x_dims[0],x_dims[1], channels]) for s in sample[0:n]]

def sample_input(gan):
    sess = gan.sess
    config = gan.config
    x = gan.graph.x
    xs = gan.graph.xs
    y = gan.graph.y
    encoded = gan.graph.x
    sample, sample2, encoded, label = sess.run([x, xs[1], encoded, y])
    return sample[0], sample2[0], encoded[0], label[0]

iteration=0
def sample(gan, sample_file):
    global iteration
    x, x2, encoded, label = sample_input(gan)
    prefix = os.path.expanduser("samples")
    sample_file = prefix+"/input-"+str(iteration)+".png"
    plot(config, x, sample_file)
    sample2_file = prefix+"/input-2-"+str(iteration)+".png"
    plot(config, x2, sample2_file)
    encoded_sample = prefix+"/encoded-"+str(iteration)+".png"
    plot(config, encoded[0], encoded_sample)

    def to_int(one_hot):
        i = 0
        for l in list(one_hot):
            if(l>0.5):
                return i
            i+=1
        return None

    sample_file = {'image':sample_file, 'label':json.dumps(to_int(label))}
    sample2_file = {'image':sample2_file, 'label':json.dumps(to_int(label))}
    encoded_sample = {'image':encoded_sample, 'label':'reconstructed'}

    sample_list = [sample_file, sample2_file, encoded_sample]
    samples = build_samples(gan)
    for s in samples:
        sample_file = prefix+"/config-"+str(iteration)+".png"
        plot(config, s, sample_file)
        sample_list.append({'image':sample_file,'label':'sample-'+str(iteration)})
        iteration += 1
    iteration += 1
    return sample_list
