# Loads an image with the tensorflow input pipeline
import glob
import os
import tensorflow as tf
from natsort import natsorted, ns
import hypergan.inputs.resize_image_patch
from tensorflow.python.ops import array_ops
from hypergan.gan_component import ValidationException, GANComponent

class MultiImageLoader:
    """
    MultiImageLoader loads a set of images into a tensorflow input pipeline.
    Supports multiple directories
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size


    def create(self, directories, channels=3, format='jpg', width=64, height=64, crop=False, resize=False, sequential=False):
        filenames_list = [natsorted(glob.glob(directory+"/*."+format)) for directory in directories]

        imgs = []

        self.datasets = []
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

        for filenames in filenames_list:
            self.file_count = len(filenames)
            filenames = tf.convert_to_tensor(filenames, dtype=tf.string)

            dataset = tf.data.Dataset.from_tensor_slices(filenames)
            if not sequential:
                print("Shuffling data")
                dataset = dataset.shuffle(self.file_count)
            dataset = dataset.map(parse_function, num_parallel_calls=4)
            dataset = dataset.batch(self.batch_size, drop_remainder=True)
            dataset = dataset.repeat()
            dataset = dataset.prefetch(1)
            shape = [self.batch_size, height, width, channels]
            self.datasets.append(tf.reshape(dataset.make_one_shot_iterator().get_next(), shape))

        self.xs = self.datasets
        self.xa = self.datasets[0]
        self.xb = self.datasets[1]
        self.x = self.datasets[0]
        return self.xs

    def inputs(self):
        return self.xs
