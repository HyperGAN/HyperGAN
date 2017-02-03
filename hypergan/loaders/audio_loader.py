import glob
import tensorflow as tf
import hypergan.loaders.resize_audio_patch
import hypergan.vendor.inception_loader as inception_loader
import hypergan.vendor.vggnet_loader as vggnet_loader
from tensorflow.contrib import ffmpeg
from hypergan.loaders import resize_audio_patch

def build_labels(dirs):
  next_id=0
  labels = {}
  for dir in dirs:
    labels[dir.split('/')[-1]]=next_id
    next_id+=1
  return labels,next_id
def mp3_tensors_from_directory(directory, batch_size, channels=2, format='mp3', seconds=30, bitrate=16384):
  path = directory+"/**/*."+format
  filenames = glob.glob(path)
  assert len(filenames)!=0, "No audio found in "+directory
  labels,total_labels = build_labels(sorted(glob.glob(directory+"/*")))
  num_examples_per_epoch = 10000

  # Create a queue that produces the filenames to read.
  classes = [labels[f.split('/')[-2]] for f in filenames]
  print("Found files", len(filenames))

  filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
  classes = tf.convert_to_tensor(classes, dtype=tf.int32)

  input_queue = tf.train.slice_input_producer([filenames, classes])

  # Read examples from files in the filename queue.
  value = tf.read_file(input_queue[0])


  label = input_queue[1]

  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  #data = tf.cast(data, tf.float32)
  data = ffmpeg.decode_audio(value, file_format=format, samples_per_second=bitrate, channel_count=channels)
  data = resize_audio_patch.resize_audio_with_crop_or_pad(data, seconds*bitrate*channels, 0,True)
  tf.Tensor.set_shape(data, [seconds*bitrate, channels])
  data = tf.reshape(data, [1, seconds*bitrate, channels])
  data = data/tf.reduce_max(tf.reshape(tf.abs(data),[-1]))
  x,y=_get_data(data, label, min_queue_examples, batch_size)

  return x, y, None, total_labels, num_examples_per_epoch


def _get_data(image, label, min_queue_examples, batch_size):
  num_preprocess_threads = 1
  images, label_batch= tf.train.shuffle_batch(
      [image, label],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity= 502,
      min_after_dequeue=128)
  return images, tf.reshape(label_batch, [batch_size])

