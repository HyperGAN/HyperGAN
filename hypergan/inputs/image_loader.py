# Loads an image with the tensorflow input pipeline
import glob
import os
import tensorflow as tf
import hypergan.inputs.resize_image_patch
from tensorflow.python.ops import array_ops
from hypergan.gan_component import ValidationException, GANComponent

class ImageLoader:
    """
    ImageLoader loads a set of images into a tensorflow input pipeline.
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_labels(self, dirs):
        total_labels=0
        labels = {}
        for dir in dirs:
            labels[dir.split('/')[-1]]=total_labels
            total_labels+=1
        if(len(dirs) == 1):
            labels = {}
            total_labels = 0
        return labels,total_labels

    def create(self, directory, channels=3, format='jpg', width=64, height=64, crop=False, resize=False):
        directories = glob.glob(directory+"/*")
        directories = [d for d in directories if os.path.isdir(d)]

        if(len(directories) == 0):
            directories = [directory] 
        labels,total_labels = self.build_labels(sorted(filter(os.path.isdir, directories)))

        # Create a queue that produces the filenames to read.
        if(len(directories) == 1):
            # No subdirectories, use all the images in the passed in path
            filenames = glob.glob(directory+"/*."+format)
            classes = [0 for f in filenames]
        else:
            filenames = glob.glob(directory+"/**/*."+format)
            classes = [labels[f.split('/')[-2]] for f in filenames]

        print("[loader] ImageLoader found", len(filenames), "images with", total_labels, "different class labels")
        self.file_count = len(filenames)
        if self.file_count == 0:
            raise ValidationException("No images found in '" + directory + "'")
        filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
        classes = tf.convert_to_tensor(classes, dtype=tf.int32)

        input_queue = tf.train.slice_input_producer([filenames, classes])

        # Read examples from files in the filename queue.
        value = tf.read_file(input_queue[0])

        if format == 'jpg':
            img = tf.image.decode_jpeg(value, channels=channels)
        elif format == 'png':
            img = tf.image.decode_png(value, channels=channels)
        else:
            print("[loader] Failed to load format", format)
        img = tf.cast(img, tf.float32)

        label = input_queue[1]

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

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4

        # Generate a batch of images and labels by building up a queue of examples.
        x,y = self._get_data(float_image, label)

        y = tf.cast(y, tf.int64)
        y = tf.one_hot(y, total_labels, 1.0, 0.0)
        self.x = x
        self.y = y
        return x, y

    def _get_data(self, image, label):
        batch_size = self.batch_size
        num_preprocess_threads = 24
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity= batch_size*10,
            min_after_dequeue=batch_size)
        return images, tf.reshape(label_batch, [batch_size])
