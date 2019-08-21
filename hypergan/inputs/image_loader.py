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

    def tfrecords_create(self, directory, channels=3, width=64, height=64, crop=False, resize=False, sequential=False):
        filenames = tf.io.gfile.glob(directory+"/*.tfrecord")
        #filenames = [directory]
        filenames = natsorted(filenames)
        print("Found tfrecord files", filenames)

            
        print("[loader] ImageLoader found", len(filenames))
        self.file_count = len(filenames)
        if self.file_count == 0:
            raise ValidationException("No images found in '" + directory + "'")
        filenames = tf.convert_to_tensor(filenames, dtype=tf.string)

        def parse_function(filename):
            def parse_record_tf(record):
                features = tf.parse_single_example(record, features={
                    'image/encoded': tf.FixedLenFeature([], tf.string)
                    #'image': tf.FixedLenFeature([], tf.string)
                    })
                #data = tf.decode_raw(features['image'], tf.uint8)
                data = tf.image.decode_jpeg(features['image/encoded'], channels=channels)
                image = tf.image.convert_image_dtype(data, dtype=tf.float32)
                image = tf.cast(data, tf.float32)* (2.0/255)-1.0
                image = tf.reshape(image, [width, height, channels])
                # Image processing for evaluation.
                # Crop the central [height, width] of the image.
                if crop:
                    image = hypergan.inputs.resize_image_patch.resize_image_with_crop_or_pad(image, height, width, dynamic_shape=True)
                elif resize:
                    image = tf.image.resize_images(image, [height, width], 1)

                tf.Tensor.set_shape(image, [height,width,channels])

                return image
            dataset = tf.data.TFRecordDataset(filename, buffer_size=8*1024*1024)
            dataset = dataset.map(parse_record_tf, num_parallel_calls=self.batch_size)

            return dataset
        def set_shape(x):
            x.set_shape(x.get_shape().merge_with(tf.TensorShape([self.batch_size, None, None, None])))
            return x
 
        # Generate a batch of images and labels by building up a queue of examples.
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        if not sequential:
            print("Shuffling data")
            dataset = dataset.shuffle(self.file_count)
        dataset = dataset.map(parse_function, num_parallel_calls=4)
        dataset = dataset.flat_map(lambda x: x.batch(self.batch_size, drop_remainder=True).repeat().prefetch(10))
        dataset = dataset.repeat().prefetch(10)
        dataset = dataset.map(set_shape)

        self.dataset = dataset
        return dataset


    def tfrecord_create(self, directory, channels=3, width=64, height=64, crop=False, resize=False, sequential=False):
        #filenames = tf.io.gfile.glob(directory+"/*.tfrecord")
        filenames = [directory]
        filenames = natsorted(filenames)
        print("Found tfrecord files", filenames)

            
        print("[loader] ImageLoader found", len(filenames))
        self.file_count = len(filenames)
        if self.file_count == 0:
            raise ValidationException("No images found in '" + directory + "'")
        filenames = tf.convert_to_tensor(filenames, dtype=tf.string)

        def parse_function(filename):
            def parse_record_tf(record):
                features = tf.parse_single_example(record, features={
                    #'image/encoded': tf.FixedLenFeature([], tf.string)
                    'image': tf.FixedLenFeature([], tf.string)
                    })
                data = tf.decode_raw(features['image'], tf.uint8)
                #data = tf.image.decode_jpeg(features['image/encoded'], channels=channels)
                image = tf.image.convert_image_dtype(data, dtype=tf.float32)
                image = tf.cast(data, tf.float32)* (2.0/255)-1.0
                image = tf.reshape(image, [width, height, channels])
                # Image processing for evaluation.
                # Crop the central [height, width] of the image.
                if crop:
                    image = hypergan.inputs.resize_image_patch.resize_image_with_crop_or_pad(image, height, width, dynamic_shape=True)
                elif resize:
                    image = tf.image.resize_images(image, [height, width], 1)

                tf.Tensor.set_shape(image, [height,width,channels])

                return image
            dataset = tf.data.TFRecordDataset(filename, buffer_size=8*1024*1024)
            dataset = dataset.map(parse_record_tf, num_parallel_calls=self.batch_size)

            return dataset
        def set_shape(x):
            x.set_shape(x.get_shape().merge_with(tf.TensorShape([self.batch_size, None, None, None])))
            return x
 
        # Generate a batch of images and labels by building up a queue of examples.
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        if not sequential:
            print("Shuffling data")
            dataset = dataset.shuffle(self.file_count)
        dataset = dataset.map(parse_function, num_parallel_calls=4)
        dataset = dataset.flat_map(lambda x: x.batch(self.batch_size, drop_remainder=True).repeat().prefetch(10))
        dataset = dataset.repeat().prefetch(10)
        dataset = dataset.map(set_shape)

        self.dataset = dataset
        return dataset


    def create(self, directory, channels=3, format='jpg', width=64, height=64, crop=False, resize=False, sequential=False):
        if format == 'tfrecord':
            return self.tfrecord_create(directory, channels=channels, width=width, height=height, crop=crop, resize=resize, sequential=sequential)
        if format == 'tfrecords':
            return self.tfrecords_create(directory, channels=channels, width=width, height=height, crop=crop, resize=resize, sequential=sequential)
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
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
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
