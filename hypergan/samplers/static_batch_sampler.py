import matplotlib.pyplot as plt
from hypergan.util.ops import *
from hypergan.util.globals import *

from hypergan.samplers.common import *

#mask_noise = None
z = None
y = None
my_iteration = 0
def build_samples(samples):
    samples = np.squeeze(samples)
    return plts

def audio_plot(size, filename, data):
    plt.clf()
    plt.figure(figsize=(2,2))
    data = np.squeeze(data)
    plt.plot(data)
    plt.xlim([0, size])
    plt.ylim([-2, 2.])
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.savefig(filename)

def sample(sample_file, sess, config):
    global z, y
    global my_iteration
    generator = get_tensor("g")[0]
    y_t = get_tensor("y")
    z_t = get_tensor("z")

    x = np.linspace(0,1, 4)

    if z is None:
        z = sess.run(z_t)
        y = sess.run(y_t)


    g=tf.get_default_graph()
    files = []
    with g.as_default():
        tf.set_random_seed(1)
        samples = sess.run(generator, feed_dict={z_t: z, y_t: y})
        i = 0
        for sample in samples[0:8]:
            i+=1
            sample_file = "samples/"+str(my_iteration)+'-'+str(i)+'.png'
            audio_plot(config['x_dims'][1], sample_file, sample)
            files.append({'image':sample_file, 'label': 'audio'})
        my_iteration += 1

    return files
