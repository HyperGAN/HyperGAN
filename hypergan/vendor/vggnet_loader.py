# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# Borrowed from tensorflow

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import op_def_registry


import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import hypergan.loaders.resize_image_patch

MODEL_DIR='/tmp/imagenet'

# pylint: disable=line-too-long
DATA_URL = 'https://github.com/pavelgonchar/colornet/blob/master/vgg/tensorflow-vgg16/vgg16-20160129.tfmodel?raw=true'
# pylint: enable=line-too-long


def create_graph(image, output_layer):
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      MODEL_DIR, 'vgg16-20160129.tfmodel?raw=true'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    for node in graph_def.node:
        print(node.name)
        #if(node.name != "DecodeJpeg" and node.name != "ResizeBilinear" and node.name != "DecodeJpeg/contents"):
        node.device = "/cpu:0"
    result= tf.import_graph_def(graph_def, name='vggnet', input_map={"images":reshape_input(image)}, return_elements=[output_layer])
    result = result[0]
    return result




def get_features(image):
    graph = create_graph(image, 'Relu_1:0')
    return tf.squeeze(graph[0])

def reshape_input(img):
  reshaped_image = tf.identity(tf.squeeze(img))
  tf.Tensor.set_shape(reshaped_image, [None, None, None])

  reshaped_image = lib.loaders.resize_image_patch.resize_image_with_crop_or_pad(reshaped_image,
                                                         224, 224, dynamic_shape=True)

  r = tf.fill([224, 224], 103.939)
  g = tf.fill([224, 224], 116.779)
  b = tf.fill([224, 224], 123.68)
  offset = tf.transpose(tf.pack([r,g,b]), [2, 1, 0])

  reshaped_image -= offset

  #reshaped_image = tf.transpose(reshaped_image, [0, 2, 1])
  reshaped_image = tf.expand_dims(reshaped_image, 0)
  print("RESHAPED", reshaped_image)

  return reshaped_image

def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = MODEL_DIR
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')


