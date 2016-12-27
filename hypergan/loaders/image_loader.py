# Loads an image with the tensorflow input pipeline
import glob
import tensorflow as tf
import hypergan.loaders.resize_image_patch
import hypergan.vendor.inception_loader as inception_loader
import hypergan.vendor.vggnet_loader as vggnet_loader

def build_labels(dirs):
  next_id=0
  labels = {}
  for dir in dirs:
    labels[dir.split('/')[-1]]=next_id
    next_id+=1
  return labels,next_id
def labelled_image_tensors_from_directory(directory, batch_size, channels=3, format='jpg', width=64, height=64, crop=True, preprocess=False):
  filenames = glob.glob(directory+"/**/*."+format)
  labels,total_labels = build_labels(sorted(glob.glob(directory+"/*")))
  num_examples_per_epoch = 30000//4
  print("[loader] ImageLoader found", len(filenames), "images with", total_labels, "different class labels")
  assert len(filenames)!=0, "No images found in "+directory

  # Create a queue that produces the filenames to read.
  classes = [labels[f.split('/')[-2]] for f in filenames]

  filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
  classes = tf.convert_to_tensor(classes, dtype=tf.int32)

  input_queue = tf.train.slice_input_producer([filenames, classes])

  # Read examples from files in the filename queue.
  value = tf.read_file(input_queue[0])
  features = tf.zeros(0)
  if(preprocess):
      f_preprocess = tf.read_file(input_queue[0]+'.preprocess')
      features = tf.decode_raw(f_preprocess, tf.float32)
      tf.Tensor.set_shape(features, [2048])

  #features = tf.identity(tf.zeros([2048]))

  if(format == 'jpg'):
      img = tf.image.decode_jpeg(value, channels=channels)
  elif(format == 'png'):
      img = tf.image.decode_png(value, channels=channels)
  else:
      print("[loader] Failed to load format", format)
  #img = tf.zeros([64,64,3])
  img = tf.cast(img, tf.float32)
  reshaped_image = tf.identity(img)
  tf.Tensor.set_shape(reshaped_image, [None, None, None])

  reshaped_image = hypergan.loaders.resize_image_patch.resize_image_with_crop_or_pad(reshaped_image,
                                                         224, 224, dynamic_shape=True)
  reshaped_image = tf.reshape(reshaped_image, [1,224,224,channels])
  #features = _get_features(reshaped_image)
  label = input_queue[1]

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  if(crop):
      resized_image = hypergan.loaders.resize_image_patch.resize_image_with_crop_or_pad(img,
                                                         height, width, dynamic_shape=True)
  else:
      resized_image = img

  #resized_image = reshaped_image
  tf.Tensor.set_shape(resized_image, [height,width,channels])
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
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  x,y,f= _get_data(float_image, label, features, min_queue_examples, batch_size)

  return x, y,f, total_labels, num_examples_per_epoch


def _get_features(image):
    vggnet_loader.maybe_download_and_extract()
    return vggnet_loader.get_features(image)

def _get_data(image, label, features, min_queue_examples, batch_size):
  num_preprocess_threads = 24
  images, label_batch, f_b= tf.train.shuffle_batch(
      [image, label, features],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity= 1000,
      min_after_dequeue=10)
  return images, tf.reshape(label_batch, [batch_size]), f_b

