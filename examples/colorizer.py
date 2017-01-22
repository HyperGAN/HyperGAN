import argparse
import tensorflow as tf
import hypergan as hg

def parse_args():
    parser = argparse.ArgumentParser(description='Train a colorizer!', add_help=True)
    parser.add_argument('directory', action='store', type=str, help='The location of your data.  Subdirectories are treated as different classes.  You must have at least 1 subdirectory.')
    parser.add_argument('--size', '-s', type=str, default='64x64x3', help='Size of your data.  For images it is widthxheightxchannels.')
    parser.add_argument('--crop', type=bool, default=False, help='If your images are perfectly sized you can skip cropping.')
    parser.add_argument('--format', '-f', type=str, default='png', help='jpg or png')
    parser.add_argument('--device', '-d', type=str, default='/gpu:0', help='In the form "/gpu:0", "/cpu:0", etc.  Always use a GPU (or TPU) to train')
    return parser.parse_args()

def add_bw(gan, net):
    bw_net = tf.image.rgb_to_grayscale(net)
    return bw_net

args = parse_args()

width = int(args.size.split("x")[0])
height = int(args.size.split("x")[1])
channels = int(args.size.split("x")[2])

config = hg.config.random(args)

#TODO add this option to D
#TODO add this option to G
config['generator.layer_filters'] = add_bw

# TODO refactor, shared in CLI
config['dtype']=tf.float32
config['batch_size'] = args.batch_size
x,y,f,num_labels,examples_per_epoch = image_loader.labelled_image_tensors_from_directory(
                        directory,
                        config['batch_size'], 
                        channels=channels, 
                        format=args.format,
                        crop=args.crop,
                        width=width,
                        height=height)

config['y_dims']=graph['num_labels']
config['x_dims']=[height,width]
config['channels']=channels

initial_graph = {
    'x':x,
    'y':y,
    'f':f,
    'num_labels':num_labels,
    'examples_per_epoch':examples_per_epoch
}


gan = GAN(config, initial_graph)

gan.load_or_initialize_graph(save_file)

tf.train.start_queue_runners(sess=gan.sess)
for i in range(100000):
    d_loss, g_loss = gan.train()

    gan.sample_to_file("samples/"+str(i)+".png")
    print("Sampled "+str(i))

tf.reset_default_graph()
self.sess.close()
