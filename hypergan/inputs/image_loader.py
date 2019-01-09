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

    def create(self, directory, channels=3, format='jpg', width=64, height=64, crop=False, resize=False, sequential=False):
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

        def parse_function(filename):
            image_string = tf.read_file(filename)
            if format == 'jpg':
                image = tf.image.decode_jpeg(image_string, channels=channels)
            elif format == 'png':
                image = tf.image.decode_png(image_string, channels=channels)
            else:
                print("[loader] Failed to load format", format)
            image = tf.cast(image, tf.float32)
            # Image processing for evaluation.
            # Crop the central [height, width] of the image.
            if crop:
                image = hypergan.inputs.resize_image_patch.resize_image_with_crop_or_pad(image, height, width, dynamic_shape=True)
            elif resize:
                image = tf.image.resize_images(image, [height, width], 1)

            image = image / 127.5 - 1.
            tf.Tensor.set_shape(image, [height,width,channels])

            return image

        # Generate a batch of images and labels by building up a queue of examples.
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        if not sequential:
            print("Shuffling data")
            dataset = dataset.shuffle(self.file_count)
        dataset = dataset.map(parse_function, num_parallel_calls=4)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(1)

        self.dataset = dataset

        self.iterator = self.dataset.make_one_shot_iterator()
        self.x = tf.reshape( self.iterator.get_next(), [self.batch_size, height, width, channels])

    def inputs(self):
        return [self.x,self.x]
