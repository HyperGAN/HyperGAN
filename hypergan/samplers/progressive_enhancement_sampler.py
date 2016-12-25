import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.globals import *
import json

# This sampler builds different images for each
# level of our GAN
# For instance, it will generate 256x256, 128x128, and 64x64 images for each z
def build_samples(sess, config):
    generator = get_tensor("g")[0]
    gs = get_tensor("gs")
    y = get_tensor("y")
    x = get_tensor("x")
    xs = get_tensor("xs")
    categories = get_tensor('categories')
    d_fake_sigmoid = get_tensor("d_fake_sigmoid")
    rand = np.random.randint(0,config['y_dims'], size=config['batch_size'])
    #rand = np.zeros_like(rand)
    random_one_hot = np.eye(config['y_dims'])[rand]
    sample, sample2, sample3, d_fake_sig = sess.run([generator, gs[1], gs[2],d_fake_sigmoid], feed_dict={y:random_one_hot})
    a = split_sample(config['sampler.samples'], d_fake_sig, sample, config['x_dims'], config['channels'])
    b = split_sample(config['sampler.samples'], d_fake_sig, sample2, [gs[1].get_shape()[1], gs[1].get_shape()[2]], config['channels'])
    c = split_sample(config['sampler.samples'], d_fake_sig, sample3, [gs[2].get_shape()[1], gs[2].get_shape()[2]], config['channels'])
    return [val for pair in zip(a, b, c) for val in pair]

def split_sample(n, d_fake_sig, sample, x_dims, channels):
    samples = []

    for s, d in zip(sample, d_fake_sig):
        samples.append({'sample':s,'d':d})
    samples = sorted(samples, key=lambda x: (1-x['d']))

    return [np.reshape(s['sample'], [x_dims[0],x_dims[1], channels]) for s in samples[0:n]]

def sample_input(sess, config):
    x = get_tensor("x")
    xs = get_tensor("xs")
    y = get_tensor("y")
    encoded = get_tensor("x")# TODO: "encoded", reuse (and encoder) not working with prelu
    sample, sample2, encoded, label = sess.run([x, xs[1], encoded, y])
    return sample[0], sample2[0], encoded[0], label[0]

iteration=0
def sample(sess, config):
    global iteration
    x, x2, encoded, label = sample_input(sess, config)
    sample_file = "samples/input-"+str(iteration)+".png"
    plot(config, x, sample_file)
    sample2_file = "samples/input-2-"+str(iteration)+".png"
    plot(config, x2, sample2_file)
    encoded_sample = "samples/encoded-"+str(iteration)+".png"
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
    samples = build_samples(sess, config)
    for s in samples:
        sample_file = "samples/config-"+str(iteration)+".png"
        plot(config, s, sample_file)
        sample_list.append({'image':sample_file,'label':'sample-'+str(iteration)})
        iteration += 1
    iteration += 1
    return sample_list
