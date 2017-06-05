# Loads an image with the tensorflow input pipeline
import glob
import os
import tensorflow as tf
import hypergan.loaders.resize_image_patch
import hypergan.vendor.inception_loader as inception_loader
import hypergan.vendor.vggnet_loader as vggnet_loader
from tensorflow.python.ops import array_ops
from hypergan.gan_component import ValidationException

class ImageLoader:

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def build_labels(self, dirs):
        next_id=0
        labels = {}
        for dir in dirs:
            labels[dir.split('/')[-1]]=next_id
            next_id+=1
        return labels,next_id

    def load(self, directory, channels=3, format='jpg', width=64, height=64, crop=False, resize=False, filterX=None):
        directories = glob.glob(directory+"/*")
        labels,total_labels = self.build_labels(sorted(filter(os.path.isdir, directories)))

        # Create a queue that produces the filenames to read.
        if filterX is not None:
            print(directories, filterX, directories[filterX])
            print("Filtering files by "+directories[filterX])
            filenames = glob.glob(directories[filterX]+"/*."+format)

        else:
            filenames = glob.glob(directory+"/**/*."+format)

        print("[loader] ImageLoader found", len(filenames), "images with", total_labels, "different class labels")
        classes = [labels[f.split('/')[-2]] for f in filenames]
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
            resized_image = hypergan.loaders.resize_image_patch.resize_image_with_crop_or_pad(img, height, width, dynamic_shape=True)
        elif resize:
            #TODO, does this add extra time if no resize happens?
            resized_image = tf.image.resize_images(img, [height, width], 1)
        else: 
            resized_image = img

        tf.Tensor.set_shape(resized_image, [height,width,channels])
        #resized_image = reshaped_image
        #resized_image = tf.image.random_flip_left_right(resized_image)
        #resized_image = tf.image.random_brightness(resized_image, 0.4)
        #resized_image = tf.image.random_contrast(resized_image, 0.2, 1.0)
        #resized_image = tf.image.random_hue(resized_image, 0.1)
        #resized_image = tf.image.random_saturation(resized_image, 0.5, 1.0)

        #resized_image = tf.image.convert_image_dtype(resized_image, tf.float32)
        # Subtract off the mean and divide by the variance of the pixels.
        #float_image = tf.image.per_image_whitening(resized_image)
        float_image = resized_image / 127.5 - 1.

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4

        # Generate a batch of images and labels by building up a queue of examples.
        x,y = self._get_data(float_image, label)

        y = tf.cast(y, tf.int64)
        y = tf.one_hot(y, total_labels, 1.0, 0.0)
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

