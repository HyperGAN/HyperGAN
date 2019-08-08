# Loads an image with the tensorflow input pipeline
import glob
import os
import tensorflow as tf
import hypergan.inputs.resize_image_patch
from tensorflow.python.ops import array_ops
from natsort import natsorted, ns
from hypergan.gan_component import ValidationException, GANComponent

class ImageLoader:
    """
    ImageLoader loads a set of images into a tensorflow input pipeline.
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def tfrecord_create(self, directory, channels=3, width=64, height=64, crop=False, resize=False, sequential=False):
        print("TFRECORD", directory)
        filenames = tf.io.gfile.glob(directory+"/*.tfrecord")
        filenames = natsorted(filenames)

            
        print("[loader] ImageLoader found", len(filenames))
        self.file_count = len(filenames)
        if self.file_count == 0:
            raise ValidationException("No images found in '" + directory + "'")
        filenames = tf.convert_to_tensor(filenames, dtype=tf.string)

        def parse_function(filename):
            def parse_record_tf(record):
                features = tf.parse_single_example(record, features={
                    'shape': tf.FixedLenFeature([3], tf.int64),
                    'data': tf.FixedLenFeature([0], tf.string)
                    })
                data = tf.decode_raw(features['data'], tf.uint8)
                data = tf.reshape(data, features['shape'])
                image = tf.cast(data, tf.float32)
                # Image processing for evaluation.
                # Crop the central [height, width] of the image.
                if crop:
                    image = hypergan.inputs.resize_image_patch.resize_image_with_crop_or_pad(image, height, width, dynamic_shape=True)
                elif resize:
                    image = tf.image.resize_images(image, [height, width], 1)

                image = image / 127.5 - 1.
                tf.Tensor.set_shape(image, [height,width,channels])

                return image
            dataset = tf.data.TFRecordDataset(filename, buffer_size=8*1024*1024)
            dataset = dataset.map(parse_record_tf, num_parallel_calls=4)

            return dataset

 
        # Generate a batch of images and labels by building up a queue of examples.
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        if not sequential:
            print("Shuffling data")
            dataset = dataset.shuffle(self.file_count)
        dataset = dataset.map(parse_function, num_parallel_calls=4)
        dataset = dataset.flat_map(lambda x: x.batch(self.batch_size))
        dataset = dataset.repeat()
        dataset = dataset.prefetch(1)

        self.dataset = dataset

        self.iterator = self.dataset.make_one_shot_iterator()
        self.x = tf.reshape( self.iterator.get_next(), [self.batch_size, height, width, channels])


    def create(self, directory, channels=3, format='jpg', width=64, height=64, crop=False, resize=False, sequential=False):
        if format == 'tfrecord':
            return self.tfrecord_create(directory, channels=channels, width=width, height=height, crop=crop, resize=resize, sequential=sequential)
        directories = glob.glob(directory+"/*")
        directories = [d for d in directories if os.path.isdir(d)]

        if(len(directories) == 0):
            directories = [directory] 

        # Create a queue that produces the filenames to read.
        if(len(directories) == 1):
            # No subdirectories, use all the images in the passed in path
            filenames = glob.glob(directory+"/*."+format)
        else:
            filenames = glob.glob(directory+"/**/*."+format)

        filenames = natsorted(filenames)

        print("[loader] ImageLoader found", len(filenames))
        self.file_count = len(filenames)
        if self.file_count == 0:
            raise ValidationException("No images found in '" + directory + "'")
        filenames = tf.convert_to_tensor(filenames, dtype=tf.string)


        # Generate a batch of images and labels by building up a queue of examples.
        dataset = tf.data.Dataset.from_tensor_slices([])
        if not sequential:
            print("Shuffling data")
            dataset = dataset.shuffle(self.file_count)
        for filen in filenames:
            dataset = dataset.concatenate(parse_function(filen))
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(1)

        self.dataset = dataset

        self.iterator = self.dataset.make_one_shot_iterator()
        self.x = tf.reshape( self.iterator.get_next(), [self.batch_size, height, width, channels])

    def inputs(self):
        return [self.x,self.x]

    def layer(self, name):
        if name == "x":
            return self.x
        return None
