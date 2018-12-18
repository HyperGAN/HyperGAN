# Loads an image with the tensorflow input pipeline
import glob
import os
import tensorflow as tf
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


    def create(self, directories, channels=3, format='jpg', width=64, height=64, crop=False, resize=False):
        directories = [d for d in directories if os.path.isdir(d)]


        filenames_list = [glob.glob(directory+"/*."+format) for directory in directories]
        [print("Found", len(filenames)) for filenames in filenames_list]

        filenames_list = [tf.convert_to_tensor(filenames, dtype=tf.string) for filenames in filenames_list]

        input_queues = [tf.train.slice_input_producer([filenames]) for filenames in filenames_list]

        imgs = []

        for input_queue in input_queues:

            # Read examples from files in the filename queue.
            value = tf.read_file(input_queue[0])

            if format == 'jpg':
                img = tf.image.decode_jpeg(value, channels=channels)
            elif format == 'png':
                img = tf.image.decode_png(value, channels=channels)
            else:
                print("[loader] Failed to load format", format)
            img = tf.cast(img, tf.float32)

          # Image processing for evaluation.
          # Crop the central [height, width] of the image.
            if crop:
                resized_image = hypergan.inputs.resize_image_patch.resize_image_with_crop_or_pad(img, height, width, dynamic_shape=True)
            elif resize:
                resized_image = tf.image.resize_images(img, [height, width], 1)
            else: 
                resized_image = img

            tf.Tensor.set_shape(resized_image, [height,width,channels])

            # This moves the image to a range of -1 to 1.
            float_image = resized_image / 127.5 - 1.

            imgs.append(float_image)

        # Generate a batch of images and labels by building up a queue of examples.
        xs = self._get_data(imgs)

        self.xs = xs
        self.xa = xs[0]
        self.xb = xs[1]
        self.x = xs[1]
        return xs

    def _get_data(self, imgs):
        batch_size = self.batch_size
        num_preprocess_threads = 24
        xs = [tf.train.shuffle_batch(
            [img],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity= batch_size*10,
            min_after_dequeue=batch_size)
            for img in imgs]
        return xs

    def inputs(self):
        return self.xs
