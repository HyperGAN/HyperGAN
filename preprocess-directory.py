
import tensorflow as tf
import shared.predata_loader
import shared.inception_loader
import argparse
from tensorflow.python.framework import ops

parser = argparse.ArgumentParser(description='This script runs inception against a directory and saves to filename.preprocessed')

parser.add_argument('--channels', type=int, default=3)
parser.add_argument('--directory', type=str)
parser.add_argument('--crop', type=bool, default=True)

parser.add_argument('--width', type=int, default=64)
parser.add_argument('--height', type=int, default=64)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--format', type=str, default='png')
parser.add_argument('--save_every', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()


import numpy as np
def save(filename, output):
    fname = filename.decode('ascii') + ".preprocess"
    print("Saving ", fname, np.min(output), np.max(output), np.mean(output))
    #output.tofile(fname)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
channels = args.channels
crop = args.crop
width = args.width
height = args.height
batch_size = 1
with tf.device("/cpu:0"):
    train_x,train_y, filename_t, num_labels,examples_per_epoch = shared.predata_loader.labelled_image_tensors_from_directory(args.directory,batch_size, channels=channels, format=args.format,crop=crop,width=width,height=height)

output_layer_t = shared.inception_loader.create_graph(train_x, "pool_3:0")
tf.train.start_queue_runners(sess=sess)

for i in range(examples_per_epoch):
    output, filename = sess.run([output_layer_t, filename_t])
    for f,o in zip(filename, output):
        #print("O IS", o.shape)
        save(f,o)
